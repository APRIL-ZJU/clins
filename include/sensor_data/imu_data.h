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

#ifndef IMU_DATA_H
#define IMU_DATA_H

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace clins {

static const double GRAVITY_NORM = -9.80;

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

struct IMUData {
  double timestamp;
  Eigen::Matrix<double, 3, 1> gyro;
  Eigen::Matrix<double, 3, 1> accel;
  SO3d orientation;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct IMUBias {
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d accel_bias;
};

struct IMUState {
  double timestamp;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Quaterniond q;
  IMUBias bias;
  Eigen::Vector3d g;
  Eigen::Quaterniond gt_q;
};

struct PoseData {
  PoseData()
      : timestamp(0),
        position(Eigen::Vector3d(0, 0, 0)),
        orientation(SO3d(Eigen::Quaterniond::Identity())) {}

  double timestamp;
  Eigen::Vector3d position;
  SO3d orientation;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline PoseData XYThetaToPoseData(double x, double y, double theta,
                                  double timestamp = 0) {
  PoseData pose;
  Eigen::Vector3d p(x, y, 0);
  Eigen::AngleAxisd rotation_vector(theta, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond q(rotation_vector);
  pose.timestamp = timestamp;
  pose.position = p;
  pose.orientation.setQuaternion(q);

  return pose;
}

inline SE3d Matrix4fToSE3d(Eigen::Matrix4f matrix) {
  Eigen::Vector3d trans(matrix(0, 3), matrix(1, 3), matrix(2, 3));
  Eigen::Quaterniond q(matrix.block<3, 3>(0, 0).cast<double>());
  q.normalize();
  return SE3d(q, trans);
}

inline void SE3dToPositionEuler(SE3d se3_pose, Eigen::Vector3d& position,
                                Eigen::Vector3d& euler) {
  position = se3_pose.translation();
  Eigen::Quaterniond q = se3_pose.unit_quaternion();
  euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
  euler *= 180 / M_PI;
}

}  // namespace clins

#endif
