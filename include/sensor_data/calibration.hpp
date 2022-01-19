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

#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#define _USE_MATH_DEFINES
#include <cmath>

#include <factor/auto_diff/imu_factor.h>
#include <factor/auto_diff/lidar_feature_factor.h>

#include <glog/logging.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/lidar_data.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <fstream>
#include <memory>
#include <utils/eigen_utils.hpp>

namespace clins {

class CalibParamManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<CalibParamManager> Ptr;

  CalibParamManager(const YAML::Node& config_node)
      : p_LinI(Eigen::Vector3d(0, 0, 0)),
        q_LtoI(Eigen::Quaterniond::Identity()),
        g_refine(Eigen::Vector2d(0, 0)),
        gyro_bias(Eigen::Vector3d(0, 0, 0)),
        acce_bias(Eigen::Vector3d(0, 0, 0)),
        gravity(Eigen::Vector3d(0, 0, GRAVITY_NORM)),
        time_offset(0) {
    // Lidar-IMU extrinsic Param
    if (config_node["extrinsic"]) {
      std::vector<double> params_vec;
      params_vec.resize(3);
      for (size_t i = 0; i < params_vec.size(); i++) {
        params_vec.at(i) = config_node["extrinsic"]["Trans"][i].as<double>();
      }
      p_LinI << params_vec[0], params_vec[1], params_vec[2];

      params_vec.resize(9);
      for (size_t i = 0; i < params_vec.size(); i++) {
        params_vec.at(i) = config_node["extrinsic"]["Rot"][i].as<double>();
      }
      Eigen::Matrix3d rot;
      rot << params_vec[0], params_vec[1], params_vec[2], params_vec[3],
          params_vec[4], params_vec[5], params_vec[6], params_vec[7],
          params_vec[8];

      q_LtoI = Eigen::Quaterniond(rot);
      q_LtoI.normalized();
      so3_LtoI = SO3d(q_LtoI);
      se3_LtoI = SE3d(so3_LtoI, p_LinI);
    }

    if (config_node["time_offset"]) {
      time_offset = config_node["time_offset"].as<double>();
    }

    if (config_node["g_refine"]) {
      std::vector<double> params_vec;
      params_vec.resize(2);
      for (size_t i = 0; i < params_vec.size(); i++) {
        params_vec.at(i) = config_node["g_refine"][i].as<double>();
      }
      g_refine << params_vec[0], params_vec[1];
    }

    /// estimate weight param
    global_opt_gyro_weight = config_node["gyro_weight"].as<double>();
    global_opt_acce_weight = config_node["accel_weight"].as<double>();
    global_opt_velocity_weight = config_node["vel_weight"].as<double>();
    global_opt_bias_weight = config_node["bias_weight"].as<double>();
    global_opt_lidar_weight = config_node["lidar_weight"].as<double>();
  }

  void ShowIMUBias() {
    std::cout << BLUE << "Gyro Bias : " << gyro_bias.transpose() << RESET
              << std::endl;
    std::cout << BLUE << "Accel Bias : " << acce_bias.transpose() << RESET
              << std::endl;
  }

  bool CheckIMUBias() {
    double gyro_threshold = 0.1;
    double acce_threshold = 0.2;
    if (fabs(gyro_bias(0)) > gyro_threshold ||
        fabs(gyro_bias(1)) > gyro_threshold ||
        fabs(gyro_bias(2)) > gyro_threshold) {
      gyro_bias = Eigen::Vector3d(0, 0, 0);
    }

    if (fabs(acce_bias(0)) > acce_threshold ||
        fabs(acce_bias(1)) > acce_threshold ||
        fabs(acce_bias(2)) > acce_threshold) {
      acce_bias = Eigen::Vector3d(0, 0, 0);
    }

    return true;
  }

  // Update after Ceres optimization
  void UpdateExtrinicParam() {
    q_LtoI = so3_LtoI.unit_quaternion();
    se3_LtoI = SE3d(so3_LtoI, p_LinI);
  }

  void UpdateGravity(Eigen::Vector3d gravity_in, int segment_id = 0) {
    gravity = gravity_in;

    gravity_in = (gravity_in / GRAVITY_NORM).eval();
    double cr = std::sqrt(gravity_in[0] * gravity_in[0] +
                          gravity_in[2] * gravity_in[2]);
    g_refine[0] = std::acos(cr);
    g_refine[1] = std::acos(-gravity_in[2] / cr);
  }

  void UpdateGravity() {
    Eigen::Map<const Eigen::Matrix<double, 2, 1>> g_param(g_refine.data());
    gravity = gravity_factor::refined_gravity<double>(g_param);
  }

  IMUBias GetIMUBias() {
    IMUBias bias;
    bias.gyro_bias = gyro_bias;
    bias.accel_bias = acce_bias;
    return bias;
  }

  Eigen::Vector3d GetGravity() { return gravity; }

  void SetGravity(Eigen::Vector3d g) {
    Eigen::Vector2d g_param = gravity_factor::recover_gravity_param(g);
    g_refine = g_param;
  }

  void SetAccelBias(Eigen::Vector3d bias) { acce_bias = bias; }

  void SetGyroBias(Eigen::Vector3d bias) { gyro_bias = bias; }

 public:
  Eigen::Vector3d p_LinI;
  SO3d so3_LtoI;
  Eigen::Quaterniond q_LtoI;
  SE3d se3_LtoI;

  Eigen::Vector2d g_refine;
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d acce_bias;
  Eigen::Vector3d gravity;

  double time_offset;

  /// opt weight
  double global_opt_gyro_weight;

  double global_opt_acce_weight;

  double global_opt_velocity_weight;

  double global_opt_bias_weight;

  double global_opt_lidar_weight;
};

}  // namespace clins

#endif
