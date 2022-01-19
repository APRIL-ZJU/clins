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

#include <sensor_data/imu_data.h>
#include <sensor_data/lidar_data.h>
#include <yaml-cpp/yaml.h>
#include <utils/eigen_utils.hpp>

namespace clins {

class OdomInitializer {
 public:
  OdomInitializer(const YAML::Node& node);

  void FeedIMUData(const IMUData& imu_data);

  bool IMUInitializer();

  inline Eigen::Vector3d GetGravity() { return gravity_; }

  inline Eigen::Vector3d GetGyroBias() { return gyro_bias_; }

  inline Eigen::Vector3d GetAccelBias() { return accel_bias_; }

  inline double GetLatestIMUTimestamp() { return imu_datas_.back().timestamp; }

  inline double GetStartIMUTimestamp() { return imu_datas_.front().timestamp; }

  inline Eigen::Quaterniond GetI0ToG() { return rot_GtoI.conjugate(); }

 private:
  Eigen::Vector3d gravity_;

  Eigen::Vector3d gyro_bias_;

  Eigen::Vector3d accel_bias_;

  double window_length_;

  double imu_excite_threshold_;

  Eigen::aligned_vector<IMUData> imu_datas_;

  Eigen::Quaterniond rot_GtoI;
};

}  // namespace clins
