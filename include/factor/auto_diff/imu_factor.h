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

#ifndef IMU_FACTOR_H
#define IMU_FACTOR_H

#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/spline/ceres_spline_helper_jet.h>
#include <basalt/spline/spline_segment.h>
#include <ceres/ceres.h>
#include <sensor_data/imu_data.h>
#include <sophus/so3.hpp>

namespace clins {
using namespace basalt;

namespace gravity_factor {

// Eigen::Matrix<ceres::Jet<double, 1>, 6, 1> gravity_jet;
template <typename T>
T gravity_jet;  //= gravity_double.template cast<T>();

template <typename T>
Eigen::Matrix<T, 3, 2> TangentBasis(
    Eigen::Map<const Eigen::Matrix<T, 3, 1>>& g0) {
  // TODO 注意这个向量方向与重力方向需要在同一半球，否则会导致 bc = 0
  Eigen::Matrix<T, 3, 1> tmp(T(0), T(0), T(-1));

  Eigen::Matrix<T, 3, 1> a = g0.normalized();
  if (a == tmp) tmp << T(-1), T(0), T(0);

  Eigen::Matrix<T, 3, 1> b = (tmp - a * (a.transpose() * tmp)).normalized();
  Eigen::Matrix<T, 3, 1> c = a.cross(b);
  Eigen::Matrix<T, 3, 2> bc;
  bc.template block<3, 1>(0, 0) = b;
  bc.template block<3, 1>(0, 1) = c;

  return bc;
}

template <typename T>
Eigen::Matrix<T, 3, 1> refined_gravity(
    Eigen::Map<const Eigen::Matrix<T, 2, 1>>& g_param) {
  T cr = ceres::cos(g_param[0]);
  T sr = ceres::sin(g_param[0]);
  T cp = ceres::cos(g_param[1]);
  T sp = ceres::sin(g_param[1]);
  return Eigen::Matrix<T, 3, 1>(-sp * cr * T(GRAVITY_NORM),
                                sr * T(GRAVITY_NORM),
                                -cr * cp * T(GRAVITY_NORM));
}

inline Eigen::Vector2d recover_gravity_param(Eigen::Vector3d& gravity) {
  Eigen::Vector2d g_param;
  double gravity_norm =
      -std::sqrt(gravity(0) * gravity(0) + gravity(1) * gravity(1) +
                 gravity(2) * gravity(2));
  double sr = gravity(1) / gravity_norm;
  g_param(0) = asin(sr);
  double sp = -gravity(0) / gravity_norm / cos(g_param(0));
  g_param(1) = asin(sp);
  return g_param;
}
}  // namespace gravity_factor

template <int _N>
class GyroscopeWithConstantBiasFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GyroscopeWithConstantBiasFactor(const IMUData& imu_data,
                                  const SplineMeta<_N>& spline_meta,
                                  double weight)
      : imu_data_(imu_data), spline_meta_(spline_meta), weight_(weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename Sophus::SO3<T>::Tangent;

    Eigen::Map<Tangent> residuals(sResiduals);

    size_t R_offset;  // should be zero if not estimate time offset
    double u;
    spline_meta_.ComputeSplineIndex(imu_data_.timestamp, R_offset, u);

    Tangent rot_vel;
    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, nullptr, &rot_vel, nullptr);

    size_t Kont_offset = spline_meta_.NumParameters();
    Eigen::Map<Tangent const> const bias(sKnots[Kont_offset]);

    residuals = rot_vel - imu_data_.gyro.template cast<T>() + bias;
    residuals = T(weight_) * residuals;

    return true;
  }

 private:
  IMUData imu_data_;
  SplineMeta<_N> spline_meta_;
  double weight_;
  double inv_dt_;
};

template <int _N>
class GyroAcceWithConstantBiasFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GyroAcceWithConstantBiasFactor(const IMUData& imu_data,
                                 const SplineMeta<_N>& spline_meta,
                                 double gyro_weight, double acce_weight)
      : imu_data_(imu_data),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        acce_weight_(acce_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    Eigen::Map<Vec6T> residuals(sResiduals);

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(imu_data_.timestamp, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_w_i;
    Tangent rot_vel;
    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T accel_w;
    CeresSplineHelper<_N>::template evaluate<T, 3, 2>(sKnots + P_offset, u,
                                                      inv_dt_, &accel_w);

    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> gyro_bias(sKnots[Kont_offset]);
    Eigen::Map<const Vec3T> acce_bias(sKnots[Kont_offset + 1]);
    Eigen::Map<const Vec2T> g_refine(sKnots[Kont_offset + 2]);

#if 1

    Vec3T gravity = gravity_factor::refined_gravity(g_refine);

#else

    Eigen::Matrix<T, 3, 2> lxly = gravity_factor::TangentBasis(gravity);
    Vec3T g_opt = (gravity + lxly * g_refine).normalized() * T(9.8);
    /// 更新重力
    gravity_factor::gg = g_opt;
    // gravity_factor::gravity_jet<T> = g_opt;
    // gravity = g_opt;
#endif
    Vec3T gyro_residuals =
        rot_vel - imu_data_.gyro.template cast<T>() + gyro_bias;
    Vec3T acce_residuals = R_w_i.inverse() * (accel_w + gravity) -
                           imu_data_.accel.template cast<T>() + acce_bias;

    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * gyro_residuals;
    residuals.template block<3, 1>(3, 0) = T(acce_weight_) * acce_residuals;

    return true;
  }

 private:
  IMUData imu_data_;
  SplineMeta<_N> spline_meta_;
  double gyro_weight_;
  double acce_weight_;
  double inv_dt_;
};

template <int _N>
class GyroAcceBiasFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GyroAcceBiasFactor(const IMUData& imu_data, const IMUBias& imu_bias,
                     const SplineMeta<_N>& spline_meta, double gyro_weight,
                     double acce_weight, double bias_weight)
      : imu_data_(imu_data),
        imu_bias_(imu_bias),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        acce_weight_(acce_weight),
        bias_weight_(bias_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using Vec12T = Eigen::Matrix<T, 12, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(imu_data_.timestamp, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_w_i;
    Tangent rot_vel;
    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T accel_w;
    CeresSplineHelper<_N>::template evaluate<T, 3, 2>(sKnots + P_offset, u,
                                                      inv_dt_, &accel_w);

    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<const Vec3T> gyro_bias(sKnots[Kont_offset]);
    Eigen::Map<const Vec3T> acce_bias(sKnots[Kont_offset + 1]);
    Eigen::Map<const Vec2T> g_refine(sKnots[Kont_offset + 2]);

    Vec3T gravity = gravity_factor::refined_gravity(g_refine);
    Vec3T gyro_residuals =
        rot_vel - imu_data_.gyro.template cast<T>() + gyro_bias;
    Vec3T acce_residuals = R_w_i.inverse() * (accel_w + gravity) -
                           imu_data_.accel.template cast<T>() + acce_bias;

    Eigen::Map<Vec12T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * gyro_residuals;
    residuals.template block<3, 1>(3, 0) = T(acce_weight_) * acce_residuals;
    residuals.template block<3, 1>(6, 0) =
        T(bias_weight_) * (gyro_bias - imu_bias_.gyro_bias.template cast<T>());
    residuals.template block<3, 1>(9, 0) =
        T(bias_weight_) * (acce_bias - imu_bias_.accel_bias.template cast<T>());

    return true;
  }

 private:
  IMUData imu_data_;
  IMUBias imu_bias_;
  SplineMeta<_N> spline_meta_;
  double gyro_weight_;
  double acce_weight_;
  double bias_weight_;
  double inv_dt_;
};

template <int _N>
class IMUPoseFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUPoseFactor(const PoseData& pose_data, const SplineMeta<_N>& spline_meta,
                double pos_weight, double rot_weight)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(pose_data_.timestamp, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_IkToG;
    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Vec3T p_IkinG;
    CeresSplineHelper<_N>::template evaluate<T, 3, 0>(sKnots + P_offset, u,
                                                      inv_dt_, &p_IkinG);

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_IkToG * pose_data_.orientation.inverse()).log();

    residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_IkinG - pose_data_.position);
    return true;
  }

 private:
  PoseData pose_data_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class IMUPositionFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUPositionFactor(const PoseData& pose_data,
                    const SplineMeta<_N>& spline_meta, double pos_weight)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(pose_data_.timestamp, P_offset, u);

    Vec3T p_IkinG;
    CeresSplineHelper<_N>::template evaluate<T, 3, 0>(sKnots + P_offset, u, 1,
                                                      &p_IkinG);

    Eigen::Map<Vec3T> residuals(sResiduals);

    residuals = T(pos_weight_) * (p_IkinG - pose_data_.position);
    return true;
  }

 private:
  PoseData pose_data_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double inv_dt_;
};

template <int _N>
class IMUOrientationFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUOrientationFactor(const PoseData& pose_data,
                       const SplineMeta<_N>& spline_meta, double rot_weight)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t R_offset;  // should be zero if not estimate time offset
    double u;
    spline_meta_.ComputeSplineIndex(pose_data_.timestamp, R_offset, u);

    SO3T R_IkToG;
    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u, 1,
                                                                 &R_IkToG);

    Eigen::Map<Tangent> residuals(sResiduals);

    residuals =
        T(rot_weight_) * ((R_IkToG * pose_data_.orientation.inverse()).log());
    return true;
  }

 private:
  PoseData pose_data_;
  SplineMeta<_N> spline_meta_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class IMUDeltaOrientationFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUDeltaOrientationFactor(const IMUData& ref_pose_data,
                            const IMUData& pose_data,
                            const SplineMeta<_N>& spline_meta,
                            double rot_weight)
      : ref_pose_data_(ref_pose_data),
        pose_data_(pose_data),
        spline_meta_(spline_meta),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    T t[2];
    t[0] = T(ref_pose_data_.timestamp);
    t[1] = T(pose_data_.timestamp);

    T u[2];
    size_t R_offset[2];

    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);

    SO3T R_IkToG[2];
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &R_IkToG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &R_IkToG[1]);

    SO3T delta_rot_est = R_IkToG[0].inverse() * R_IkToG[1];
    SO3T delta_rot_mea =
        ref_pose_data_.orientation.inverse() * pose_data_.orientation;

    Eigen::Map<Tangent> residuals(sResiduals);

    residuals =
        T(rot_weight_) * ((delta_rot_mea.inverse() * delta_rot_est).log());
    return true;
  }

 private:
  IMUData ref_pose_data_;
  IMUData pose_data_;
  SplineMeta<_N> spline_meta_;
  double rot_weight_;
  double inv_dt_;
};

/// IMU的速度表示在全局坐标系下
template <int _N>
class IMUGlobalVelocityFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUGlobalVelocityFactor(const Eigen::Vector3d& velocity,
                          const double timestamp,
                          const SplineMeta<_N>& spline_meta, double vel_weight)
      : velocity_(velocity),
        timestamp_(timestamp),
        spline_meta_(spline_meta),
        vel_weight_(vel_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    size_t P_offset;
    double u;
    spline_meta_.ComputeSplineIndex(timestamp_, P_offset, u);

    Vec3T v_IkinG;
    CeresSplineHelper<_N>::template evaluate<T, 3, 1>(sKnots + P_offset, u,
                                                      inv_dt_, &v_IkinG);

    Eigen::Map<Vec3T> residuals(sResiduals);

    residuals = T(vel_weight_) * (v_IkinG - velocity_.template cast<T>());
    return true;
  }

 private:
  Eigen::Vector3d velocity_;
  double timestamp_;
  SplineMeta<_N> spline_meta_;
  double vel_weight_;
  double inv_dt_;
};

}  // namespace clins

#endif
