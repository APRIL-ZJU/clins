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

#include <odometry/inertial_initializer.h>

namespace clins {

OdomInitializer::OdomInitializer(const YAML::Node &node)
    : gravity_(Eigen::Vector3d(0, 0, 9.81)),
      gyro_bias_(Eigen::Vector3d(0, 0, 0)),
      accel_bias_(Eigen::Vector3d(0, 0, 0)),
      window_length_(1.0),
      imu_excite_threshold_(1.0) {
  if (node["window_length"]) {
    window_length_ = node["window_length"].as<double>();
  }
  if (node["imu_excite_threshold"]) {
    imu_excite_threshold_ = node["imu_excite_threshold"].as<double>();
  }
  if (node["gravity"]) {
    std::vector<double> params_vec;
    params_vec.resize(3);
    for (size_t i = 0; i < params_vec.size(); i++) {
      params_vec.at(i) = node["gravity"][i].as<double>();
    }
    gravity_ << params_vec[0], params_vec[1], params_vec[2];
  }
}

void OdomInitializer::FeedIMUData(const IMUData &imu_data) {
  imu_datas_.push_back(imu_data);

  auto it0 = imu_datas_.begin();
  while (it0 != imu_datas_.end() &&
         it0->timestamp < imu_data.timestamp - 2 * window_length_) {
    it0 = imu_datas_.erase(it0);
  }
}

bool OdomInitializer::IMUInitializer() {
  if (imu_datas_.empty()) return false;

  if (imu_datas_.back().timestamp - imu_datas_.begin()->timestamp <
      window_length_)
    return false;

  Eigen::Vector3d accel_avg(0, 0, 0);
  Eigen::Vector3d gyro_avg(0, 0, 0);

  std::vector<IMUData> imu_cache;
  for (size_t i = imu_datas_.size() - 1; i >= 0; i--) {
    accel_avg += imu_datas_[i].accel;
    gyro_avg += imu_datas_[i].gyro;
    imu_cache.push_back(imu_datas_[i]);
    if (imu_datas_.back().timestamp - imu_datas_[i].timestamp >= window_length_)
      break;
  }
  accel_avg /= (int)imu_cache.size();
  gyro_avg /= (int)imu_cache.size();

  double accel_var = 0;
  for (size_t i = 0; i < imu_cache.size(); i++) {
    accel_var +=
        (imu_cache[i].accel - accel_avg).dot(imu_cache[i].accel - accel_avg);
  }
  accel_var = std::sqrt(accel_var / ((int)imu_cache.size() - 1));

  if (accel_var >= imu_excite_threshold_) {
    return false;
  }

  Eigen::Vector3d z_axis = accel_avg / accel_avg.norm();
  Eigen::Vector3d e_1(1, 0, 0);
  Eigen::Vector3d x_axis = e_1 - z_axis * z_axis.transpose() * e_1;
  x_axis = x_axis / x_axis.norm();

  Eigen::Matrix<double, 3, 1> y_axis =
      Eigen::SkewSymmetric<double>(z_axis) * x_axis;

  Eigen::Matrix<double, 3, 3> Rot;
  Rot.block(0, 0, 3, 1) = x_axis;
  Rot.block(0, 1, 3, 1) = y_axis;
  Rot.block(0, 2, 3, 1) = z_axis;

  rot_GtoI = Eigen::Quaterniond(Rot);

  accel_bias_ = accel_avg - rot_GtoI.toRotationMatrix() * gravity_;
  gyro_bias_ = gyro_avg;

  return true;
}

}  // namespace clins
