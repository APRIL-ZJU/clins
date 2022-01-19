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

#ifndef LOOP_CLOSURE_FACTOR_H
#define LOOP_CLOSURE_FACTOR_H

#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/spline/ceres_spline_helper_jet.h>
#include <basalt/spline/spline_segment.h>
#include <ceres/ceres.h>
#include <sensor_data/imu_data.h>
#include <sensor_data/loop_closure_data.h>
#include <sophus/so3.hpp>

namespace clins {
using namespace basalt;

template <int _N>
class RelativeTrajectoryPoseFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativeTrajectoryPoseFactor(const RelativePoseData& measurement,
                               const SplineMeta<_N>& spline_meta,
                               double pos_weight, double rot_weight)
      : measurement_(measurement),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;

    T t[2];
    t[0] = T(measurement_.target_timestamp);
    t[1] = T(measurement_.source_timestamp);

    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);
    P_offset[1] = R_offset[1] + spline_meta_.NumParameters();

    SO3T target_R_ItoG, source_R_ItoG;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &target_R_ItoG);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &source_R_ItoG);

    Vec3T target_p_IinG, source_p_IinG;
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[0], u[0], inv_dt_, &target_p_IinG);
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[1], u[1], inv_dt_, &source_p_IinG);

    SO3T R_StoT = target_R_ItoG.inverse() * source_R_ItoG;
    Vec3T p_SinT = target_R_ItoG.inverse() * (source_p_IinG - target_p_IinG);

    Eigen::Map<Vec6T> residuals(sResiduals);

    residuals.template block<3, 1>(0, 0) =
        T(pos_weight_) * (p_SinT - measurement_.position);
    residuals.template block<3, 1>(3, 0) =
        T(rot_weight_) * (R_StoT * measurement_.orientation.inverse()).log();

    return true;
  }

 private:
  RelativePoseData measurement_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class LoopClosureEdgesFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LoopClosureEdgesFactor(const RelativePoseData& measurement,
                         const SplineMeta<_N>& spline_meta, double pos_weight,
                         double rot_weight)
      : measurement_(measurement),
        spline_meta_(spline_meta),
        pos_weight_(pos_weight),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;

    T t[2];
    t[0] = T(measurement_.target_timestamp);
    t[1] = T(measurement_.source_timestamp);

    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);
    P_offset[1] = R_offset[1] + spline_meta_.NumParameters();

    SO3T target_R_ItoG, source_R_ItoG;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &target_R_ItoG);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &source_R_ItoG);

    Vec3T target_p_IinG, source_p_IinG;
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[0], u[0], inv_dt_, &target_p_IinG);
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[1], u[1], inv_dt_, &source_p_IinG);

    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_LtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_LinI(sKnots[Kont_offset + 1]);

    SO3T target_R_LtoG = target_R_ItoG * R_LtoI;
    Vec3T target_p_LinG = target_R_ItoG * p_LinI + target_p_IinG;

    SO3T source_R_LtoG = source_R_ItoG * R_LtoI;
    Vec3T source_p_LinG = source_R_ItoG * p_LinI + source_p_IinG;

    SO3T R_StoT = target_R_LtoG.inverse() * source_R_LtoG;
    Vec3T p_SinT = target_R_LtoG.inverse() * (source_p_LinG - target_p_LinG);

    Eigen::Map<Vec6T> residuals(sResiduals);

    residuals.template block<3, 1>(0, 0) =
        T(pos_weight_) * (p_SinT - measurement_.position);
    residuals.template block<3, 1>(3, 0) =
        T(rot_weight_) * (R_StoT * measurement_.orientation.inverse()).log();

    return true;
  }

 private:
  RelativePoseData measurement_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class VelocityConstraintFactor : public CeresSplineHelper<_N> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VelocityConstraintFactor(const VelocityData& velocity_data,
                           const SplineMeta<_N>& spline_meta,
                           double gyro_weight, double velocity_weight)
      : velocity_data_(velocity_data),
        spline_meta_(spline_meta),
        gyro_weight_(gyro_weight),
        velocity_weight_(velocity_weight) {
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
    spline_meta_.ComputeSplineIndex(velocity_data_.timestamp, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_w_i;
    Tangent rot_vel;
    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_w_i, &rot_vel);

    Vec3T vel_w;
    CeresSplineHelper<_N>::template evaluate<T, 3, 1>(sKnots + P_offset, u,
                                                      inv_dt_, &vel_w);

    Vec3T gyro_residuals = rot_vel - velocity_data_.gyro.template cast<T>();
    Vec3T vel_residuals =
        R_w_i.inverse() * vel_w - velocity_data_.velocity.template cast<T>();

    residuals.template block<3, 1>(0, 0) = T(gyro_weight_) * gyro_residuals;
    residuals.template block<3, 1>(3, 0) = T(velocity_weight_) * vel_residuals;

    return true;
  }

 private:
  VelocityData velocity_data_;
  SplineMeta<_N> spline_meta_;
  double gyro_weight_;
  double velocity_weight_;
  double inv_dt_;
};

}  // namespace clins

#endif
