/*
 * CLINS: Continuous-Time Trajectory Estimation for LiDAR-Inertial System
 * Copyright (C) 2022 Jiajun Lv
 * Copyright (C) 2022 Kewei Hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TRAJECTORY_MANAGER_HPP
#define TRAJECTORY_MANAGER_HPP

#include <odometry/imu_state_estimator.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/loop_closure_data.h>
#include <utils/tic_toc.h>
#include <odometry/lidar_odometry.hpp>
#include <sensor_data/calibration.hpp>
#include <trajectory/se3_trajectory.hpp>
#include <trajectory/trajectory_estimator.hpp>
#include <trajectory/trajectory_viewer.hpp>

#include <pcl/kdtree/impl/kdtree_flann.hpp>

namespace clins {

template <int _N>
class TrajectoryManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using TrajectoryN = Trajectory<_N>;

  typedef std::shared_ptr<TrajectoryManager<_N>> Ptr;

  TrajectoryManager(std::shared_ptr<TrajectoryN> trajectory,
                    std::shared_ptr<CalibParamManager> calib_param,
                    std::shared_ptr<ImuStateEstimator> state_estimator)
      : trajectory_(trajectory),
        calib_param_(calib_param),
        imu_state_estimator_(state_estimator) {}

  void SetTrajectory(std::shared_ptr<TrajectoryN> trajectory) {
    trajectory_ = trajectory;
  }

  void AddIMUData(IMUData data) {
    data.timestamp -= trajectory_->GetDataStartTime();
    imu_data_.emplace_back(data);
    if (trajectory_init_) {
      imu_state_estimator_->FeedIMUData(data);
      cache_imu_data_.emplace_back(data);
    }
  }

  void InitIMUData(double feature_time) {
    double traj_start_time;
    for (size_t i = imu_data_.size() - 1; i >= 0; i--) {
      if (imu_data_[i].timestamp <= feature_time) {
        traj_start_time = imu_data_[i].timestamp;
        break;
      }
    }

    /// remove imu data
    auto iter = imu_data_.begin();
    while (iter != imu_data_.end()) {
      if (iter->timestamp < traj_start_time) {
        iter = imu_data_.erase(iter);
      } else {
        iter->timestamp -= traj_start_time;
        iter++;
      }
    }

    //    std::cout << "IMU data first timestamp : " <<
    //    imu_data_.front().timestamp
    //              << std::endl;
    trajectory_->setDataStartTime(traj_start_time);
    imu_state_estimator_->SetFirstIMU(imu_data_.front());
    for (size_t i = 1; i < imu_data_.size(); i++) {
      imu_state_estimator_->FeedIMUData(imu_data_[i]);
    }
    trajectory_init_ = true;
  }

  std::shared_ptr<TrajectoryN> get_trajectory() { return trajectory_; }

  bool BuildProblemAndSolve(
      const Eigen::aligned_vector<PointCorrespondence>& point_measurement,
      int iteration = 100);

  void UpdateTrajectoryProperty();

  void IntegrateIMUMeasurement(double scan_min, double scan_max);

  void EstimateIMUBias();

  void OptimiseLoopClosure(const LoopClosureOptimizationParam& param,
                           const LoopClosureWeights& weights);

  void SetInitialPoseRotation(Eigen::Quaterniond q) {
    init_pose.orientation.setQuaternion(q);
  }

  void SetUseCornerFeature(bool use_corner) {
    use_corner_feature_ = use_corner;
  }

  void SetUseIMUOrientation(bool use_orientation) {
    use_imu_orientation_ = use_orientation;
  }

  void PlotIMUMeasurement(std::string cache_path) {
    TrajectoryViewer::PublishIMUData<_N>(trajectory_, cache_imu_data_,
                                         cache_imu_bias_, cache_path);
  }

 private:
  void SetPriorCorrespondence();

  void AddStartTimePose(std::shared_ptr<TrajectoryEstimator<_N>> estimator);

  std::shared_ptr<TrajectoryN> trajectory_;

  CalibParamManager::Ptr calib_param_;

  std::shared_ptr<ImuStateEstimator> imu_state_estimator_;

  Eigen::aligned_vector<PointCorrespondence> point_prior_database_;
  Eigen::aligned_vector<IMUData> imu_data_;
  Eigen::aligned_vector<IMUData> cache_imu_data_;
  std::vector<std::pair<double, IMUBias>> cache_imu_bias_;

  double cor_min_time_, cor_max_time_;

  bool trajectory_init_ = false;

  PoseData init_pose;

  bool use_corner_feature_ = true;
  bool use_imu_orientation_ = false;
};

template <int _N>
void TrajectoryManager<_N>::SetPriorCorrespondence() {
  double prior_time = trajectory_->GetActiveTime();
  if (prior_time <= trajectory_->minTime()) {
    return;
  }

  // https://stackoverflow.com/questions/991335/
  for (auto iter = imu_data_.begin(); iter != imu_data_.end();) {
    if (iter->timestamp < prior_time) {
      iter = imu_data_.erase(iter);
    } else {
      ++iter;
    }
  }
}

template <int _N>
void TrajectoryManager<_N>::AddStartTimePose(
    std::shared_ptr<TrajectoryEstimator<_N>> estimator) {
  size_t kont_idx = trajectory_->computeTIndex(cor_min_time_).second;
  if (kont_idx < _N) {
    init_pose.timestamp = trajectory_->minTime();

    double rot_weight = 100;
    double pos_weight = 100;
    estimator->AddPoseMeasurement(init_pose, rot_weight, pos_weight);
  }
}

template <int _N>
bool TrajectoryManager<_N>::BuildProblemAndSolve(
    const Eigen::aligned_vector<PointCorrespondence>& point_measurement,
    int iteration) {
  if (point_measurement.empty() || imu_data_.empty()) {
    std::cout << "[BuildProblemAndSolve] input empty data "
              << point_measurement.size() << ", " << imu_data_.size()
              << std::endl;
    return false;
  }

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);


  double feature_weight = calib_param_->global_opt_lidar_weight;
  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double bias_weight = calib_param_->global_opt_bias_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  for (const auto& v : point_measurement) {
    estimator->AddLoamMeasurement(v, feature_weight);
  }

  IMUBias imu_bias = calib_param_->GetIMUBias();
  for (const auto& v : imu_data_) {
    if (v.timestamp >= cor_max_time_) break;
    //    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
    estimator->AddIMUBiasMeasurement(v, imu_bias, gyro_weight, accel_weight,
                                     bias_weight);
  }

  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cor_max_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }

  //  double lambda = 1.0;
  //  Eigen::Matrix3d a2_weight = lambda * Eigen::Matrix3d::Identity();
  //  estimator->AddQuadraticIntegralFactor(cor_max_time_,
  //  trajectory_->maxTime()-1e-9, a2_weight); std::cout << "motion weight: " <<
  //  lambda << std::endl;

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, 1);
  }

  AddStartTimePose(estimator);


  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true); 

  // estimator->AddCallback(cor_min_time_);
  ceres::Solver::Summary summary = estimator->Solve(iteration, false);

  calib_param_->CheckIMUBias();
  return true;
}

template <int _N>
void TrajectoryManager<_N>::UpdateTrajectoryProperty() {
  trajectory_->UpdateActiveTime(cor_max_time_);
  trajectory_->SetForcedFixedTime(cor_min_time_ - 0.1);

  TrajectoryViewer::PublishSplineTrajectory<4>(
      trajectory_, trajectory_->minTime(), cor_max_time_, 0.02);

  cache_imu_bias_.push_back(
      std::make_pair(cor_min_time_, calib_param_->GetIMUBias()));
}

template <int _N>
void TrajectoryManager<_N>::IntegrateIMUMeasurement(double scan_time_min,
                                                    double scan_time_max) {
  if (imu_data_.empty()) {
    std::cout << "[IntegrateIMUMeasurement] IMU data empty! " << std::endl;
    return;
  }

  cor_min_time_ = scan_time_min;
  cor_max_time_ = scan_time_max;

  imu_state_estimator_->GetLatestIMUState<_N>(trajectory_);
  imu_state_estimator_->Propagate(scan_time_max);

  SetPriorCorrespondence();

  SE3d last_kont = trajectory_->getLastKnot();
  trajectory_->extendKnotsTo(scan_time_max, last_kont);

  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;
  double vel_weight = calib_param_->global_opt_velocity_weight;

  estimator->LockTrajectory(false);
  if (trajectory_->GetForcedFixedTime() > 0) {
    estimator->SetKeyScanConstant(trajectory_->GetForcedFixedTime());
  }

  for (const auto& state : imu_state_estimator_->GetIntegrateState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  for (const auto& v : imu_data_) {
    if (v.timestamp >= scan_time_max) break;
    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
  }

  if (use_imu_orientation_) {
    if (imu_data_.size() > 2) {
      IMUData ref_imu_data = imu_data_.front();
      for (size_t i = 1; i < imu_data_.size(); i++) {
        if (imu_data_[i].timestamp >= cor_max_time_) break;
        estimator->AddIMUDeltaOrientationMeasurement(ref_imu_data, imu_data_[i],
                                                     gyro_weight);
      }
    }
  }

  //  double lambda = 1.0;
  //  Eigen::Matrix3d a2_weight = lambda * Eigen::Matrix3d::Identity();
  //  estimator->AddQuadraticIntegralFactor(
  //      scan_time_max, trajectory_->maxTime() - 1e-9, a2_weight);

  imu_state_estimator_->Predict(trajectory_->maxTime() - 1e-9, 0.01);

  for (const auto& state : imu_state_estimator_->GetVirtualIMUState()) {
    estimator->AddGlobalVelocityMeasurement(state, vel_weight);
  }

  AddStartTimePose(estimator);

  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(true, true, true);

  ceres::Solver::Summary summary = estimator->Solve(50, false);

  //  TrajectoryViewer::PublishIMUData<4>(trajectory_, imu_data_);
  TrajectoryViewer::PublishSplineTrajectory<4>(
      trajectory_, trajectory_->minTime(), scan_time_max, 0.02);
}

template <int _N>
void TrajectoryManager<_N>::EstimateIMUBias() {
  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  estimator->LockTrajectory(true);
  double gyro_weight = calib_param_->global_opt_gyro_weight;
  double accel_weight = calib_param_->global_opt_acce_weight;

  for (const auto& v : imu_data_) {
    if (v.timestamp >= cor_max_time_) break;
    estimator->AddIMUMeasurement(v, gyro_weight, accel_weight);
  }
  estimator->LockExtrinsicParam(true, true);
  estimator->LockIMUState(false, false, true);

  ceres::Solver::Summary summary = estimator->Solve(50, true);
  //  calib_param_->ShowIMUBias();
}

template <int _N>
void TrajectoryManager<_N>::OptimiseLoopClosure(
    const LoopClosureOptimizationParam& param,
    const LoopClosureWeights& weights) {
  std::shared_ptr<TrajectoryEstimator<_N>> estimator =
      std::make_shared<TrajectoryEstimator<_N>>(trajectory_, calib_param_);

  estimator->LockTrajectory(false);
  estimator->SetKeyScanConstant(param.history_key_frame_max_time);

  for (const auto& v : param.velocity_constraint) {
    estimator->AddVelocityConstraintMeasurement(v, weights.velocity_weight,
                                                weights.gyro_weight);
  }

  for (const auto& p : param.pose_graph_key_pose) {
    estimator->AddLiDARPoseMeasurement(p, weights.pose_graph_edge_rot_weight,
                                       weights.pose_graph_edge_pos_weight);
  }

  estimator->LockExtrinsicParam(true, true);

  ceres::Solver::Summary summary = estimator->Solve(100, true);

  TrajectoryViewer::PublishSplineTrajectory<4>(
      trajectory_, trajectory_->minTime(), trajectory_->maxTime(), 0.02);
}

}  // namespace clins

#endif
