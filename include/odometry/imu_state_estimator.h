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

#ifndef IMU_STATE_ESTIMATOR_H
#define IMU_STATE_ESTIMATOR_H

#include <sensor_data/imu_data.h>
#include <yaml-cpp/yaml.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <trajectory/se3_trajectory.hpp>

namespace clins {

enum MotionState {
  start_motionless = 0,  /// Initialization
  moving,                ///
};

class ImuStateEstimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<ImuStateEstimator> Ptr;

  ImuStateEstimator(const YAML::Node& node);

  void FeedIMUData(const IMUData& imu_data);

  void SetFirstIMU(IMUData imu) {
    latest_imu_ = imu;
    first_imu_orientation_inv_ = imu.orientation.inverse().unit_quaternion();
  }

  template <int _N>
  void GetLatestIMUState(std::shared_ptr<Trajectory<_N>> trajectory) {
    Eigen::Vector3d velocity_world =
        trajectory->transVelWorld(latest_imu_time_);
    Sophus::SO3d R = trajectory->pose(latest_imu_time_).so3();
    Eigen::Vector3d P = trajectory->pose(latest_imu_time_).translation();
    latest_state_.timestamp = latest_imu_time_;
    latest_state_.p = P;
    latest_state_.q = R.unit_quaternion();
    latest_state_.v = velocity_world;
    latest_state_.bias = trajectory->GetCalibParam()->GetIMUBias();
    latest_state_.g = trajectory->GetCalibParam()->GetGravity();
  }

  void Propagate(double timestamp);

  void Predict(double trajecroty_max_time, double sample_time);

  void UpdateMotionState(Eigen::aligned_vector<IMUData>& imu_cache);

  Eigen::aligned_vector<IMUState> GetIntegrateState() {
    return integrate_imu_state_;
  }

  Eigen::aligned_vector<IMUState> GetVirtualIMUState() {
    return virtual_imu_state_;
  }

 private:
  Eigen::aligned_vector<IMUData> imu_data_;

  /// in global frame
  IMUState latest_state_;

  double latest_imu_time_;

  IMUData latest_imu_;

  Eigen::aligned_vector<IMUState> integrate_imu_state_;

  Eigen::aligned_vector<IMUState> virtual_imu_state_;

  MotionState motion_state_;

  double accel_excite_threshold_;

  double gyro_excite_threshold_;

  int sample_num_;

  Eigen::Quaterniond first_imu_orientation_inv_;
};

}  // namespace clins

#endif
