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

#include <odometry/imu_state_estimator.h>

namespace clins {

ImuStateEstimator::ImuStateEstimator(const YAML::Node& node)
    : latest_imu_time_(0),
      motion_state_(MotionState::start_motionless),
      accel_excite_threshold_(0.5),
      gyro_excite_threshold_(0.5),
      sample_num_(5) {
  if (node["accel_excite_threshold"]) {
    accel_excite_threshold_ = node["accel_excite_threshold"].as<double>();
  }
  if (node["gyro_excite_threshold"]) {
    gyro_excite_threshold_ = node["gyro_excite_threshold"].as<double>();
  }
  if (node["sample_num"]) {
    sample_num_ = node["sample_num"].as<double>();
  }
}

void ImuStateEstimator::FeedIMUData(const IMUData& imu_data) {
  imu_data_.push_back(imu_data);
}

void ImuStateEstimator::Propagate(double timestamp) {
  if (imu_data_.empty()) {
    std::cout << RED << "[ImuStateEstimator] no imu data !" << RESET
              << std::endl;
  }
  Eigen::aligned_vector<IMUData> imu_cache;
  auto iter = imu_data_.begin();
  while (iter != imu_data_.end()) {
    if (iter->timestamp < timestamp) {
      imu_cache.push_back(*iter);
      iter = imu_data_.erase(iter);
    } else
      break;
  }

  //  std::cout << "IMU cache size : " << imu_cache.size() << " : " << timestamp
  //  << std::endl;
  integrate_imu_state_.clear();

  UpdateMotionState(imu_cache);
  if (motion_state_ == MotionState::start_motionless) {
    for (size_t i = 0; i < imu_cache.size(); i += sample_num_) {
      IMUState predict_state = latest_state_;
      predict_state.timestamp = imu_cache[i].timestamp;
      predict_state.gt_q = first_imu_orientation_inv_ *
                           imu_cache[i].orientation.unit_quaternion();
      integrate_imu_state_.push_back(predict_state);
    }
    latest_imu_time_ = imu_cache.back().timestamp;
    latest_imu_ = imu_cache.back();
  } else if (motion_state_ == MotionState::moving) {
    for (size_t i = 0; i < imu_cache.size(); i++) {
      double dt = imu_cache[i].timestamp - latest_imu_.timestamp;
      latest_state_.timestamp = imu_cache[i].timestamp;
      Eigen::Vector3d un_acc_0 =
          latest_state_.q *
              (latest_imu_.accel - latest_state_.bias.accel_bias) -
          latest_state_.g;
      Eigen::Vector3d un_gyro = 0.5 * (latest_imu_.gyro + imu_cache[i].gyro) -
                                latest_state_.bias.gyro_bias;
      latest_state_.q *= Eigen::Quaterniond(
          1, un_gyro(0) * dt / 2, un_gyro(1) * dt / 2, un_gyro(2) * dt / 2);
      Eigen::Vector3d un_acc_1 =
          latest_state_.q *
              (imu_cache[i].accel - latest_state_.bias.accel_bias) -
          latest_state_.g;
      Eigen::Vector3d un_accel = 0.5 * (un_acc_0 + un_acc_1);
      latest_state_.p += latest_state_.v * dt + 0.5 * un_accel * dt * dt;
      latest_state_.v += un_accel * dt;

      latest_state_.gt_q = first_imu_orientation_inv_ *
                           imu_cache[i].orientation.unit_quaternion();

      if (i % sample_num_ == 0) {
        integrate_imu_state_.push_back(latest_state_);
      }
      latest_imu_time_ = imu_cache.back().timestamp;
      latest_imu_ = imu_cache[i];
    }
  }
}

void ImuStateEstimator::Predict(double trajecroty_max_time,
                                double sample_time) {
  virtual_imu_state_.clear();
  double predict_start_time = latest_state_.timestamp + sample_time;
  for (double t = predict_start_time; t < trajecroty_max_time;
       t += sample_time) {
    latest_state_.timestamp = t;
    virtual_imu_state_.push_back(latest_state_);
  }
  latest_state_.timestamp = trajecroty_max_time;
  virtual_imu_state_.push_back(latest_state_);
}

void ImuStateEstimator::UpdateMotionState(
    Eigen::aligned_vector<IMUData>& imu_cache) {
  Eigen::Vector3d accel_avg(0, 0, 0);
  Eigen::Vector3d gyro_avg(0, 0, 0);
  for (size_t i = 0; i < imu_cache.size(); i++) {
    accel_avg += imu_cache[i].accel;
    gyro_avg += imu_cache[i].gyro;
  }
  accel_avg /= (int)imu_cache.size();
  gyro_avg /= (int)imu_cache.size();

  double accel_var = 0;
  double gyro_var = 0;
  for (size_t i = 0; i < imu_cache.size(); i++) {
    accel_var +=
        (imu_cache[i].accel - accel_avg).dot(imu_cache[i].accel - accel_avg);
    gyro_var +=
        (imu_cache[i].gyro - gyro_avg).dot(imu_cache[i].gyro - gyro_avg);
  }
  accel_var = std::sqrt(accel_var / ((int)imu_cache.size() - 1));
  gyro_var = std::sqrt(gyro_var / ((int)imu_cache.size() - 1));

  //  std::cout << GREEN << "[UpdateMotionState] accel_var : " << accel_var
  //            << " gyro_var : " << gyro_var << RESET << std::endl;

  /// TODO
  if (accel_var >= accel_excite_threshold_) {
    if (motion_state_ == MotionState::start_motionless) {
      motion_state_ = MotionState::moving;
    }
  }
}

}  // namespace clins
