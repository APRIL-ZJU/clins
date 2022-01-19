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

#ifndef AUTO_DIFF_LIDAR_FEATURE_FACTOR
#define AUTO_DIFF_LIDAR_FEATURE_FACTOR

#include <basalt/spline/ceres_spline_helper_jet.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <feature/lidar_feature.h>
#include <sensor_data/imu_data.h>
#include <Eigen/Core>
#include <memory>
#include <sophus/so3.hpp>

namespace clins {

using namespace basalt;

template <int _N>
class PointFeatureFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointFeatureFactor(const PointCorrespondence& pc,
                     const SplineMeta<_N>& spline_meta, double weight)
      : measurement_(pc), spline_meta_(spline_meta), weight_(weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using SO3T = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    T t[2];
    t[0] = T(measurement_.t_map);
    t[1] = T(measurement_.t_point);

    T u[2];
    size_t R_offset[2];
    size_t P_offset[2];
    spline_meta_.ComputeSplineIndex(t[0], R_offset[0], u[0]);
    P_offset[0] = R_offset[0] + spline_meta_.NumParameters();

    spline_meta_.ComputeSplineIndex(t[1], R_offset[1], u[1]);
    P_offset[1] = R_offset[1] + spline_meta_.NumParameters();

    SO3T R_IkToG[2];
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[0], u[0], inv_dt_, &R_IkToG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset[1], u[1], inv_dt_, &R_IkToG[1]);

    Vec3T p_IkinG[2];
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[0], u[0], inv_dt_, &p_IkinG[0]);
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(
        sKnots + P_offset[1], u[1], inv_dt_, &p_IkinG[1]);

    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_LtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_LinI(sKnots[Kont_offset + 1]);

    SO3T R_IkToI0 = R_IkToG[0].inverse() * R_IkToG[1];
    Vec3T p_IkinI0 = R_IkToG[0].inverse() * (p_IkinG[1] - p_IkinG[0]);

    // {M} frame is coinside with {L0) frame.
    Vec3T p_Lk = measurement_.point.template cast<T>();
    Vec3T p_Ik = R_LtoI * p_Lk + p_LinI;
    Vec3T p_I0 = R_IkToI0 * p_Ik + p_IkinI0;
    Vec3T p_L0 = R_LtoI.inverse() * (p_I0 - p_LinI);

    T dist;
    if (measurement_.geo_type == GeometryType::Plane) {
      Vec3T norm = (measurement_.geo_plane.template cast<T>()).head(3);
      dist = p_L0.dot(norm) + T(measurement_.geo_plane[3]);
    } else {
      // omit item 1 =: 1.0 / measurement_.geo_normal.norm()
      dist = (p_L0 - measurement_.geo_point.template cast<T>())
                 .cross(measurement_.geo_normal.template cast<T>())
                 .norm();
    }

    Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(sResiduals);
    residuals.template block<1, 1>(0, 0) = Eigen::Matrix<T, 1, 1>(dist);

    residuals = T(weight_) * residuals;

    return true;
  }

 private:
  PointCorrespondence measurement_;
  SplineMeta<_N> spline_meta_;
  double weight_;
  double inv_dt_;
};

template <int _N>
class LiDARPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LiDARPoseFactor(const PoseData& pose_data, const SplineMeta<_N>& spline_meta,
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

    size_t Kont_offset = 2 * spline_meta_.NumParameters();

    T t_offset = sKnots[Kont_offset + 2][0];
    T t = T(pose_data_.timestamp) + t_offset;

    size_t R_offset;  // should be zero if not estimate time offset
    size_t P_offset;
    T u;
    spline_meta_.ComputeSplineIndex(t, R_offset, u);
    P_offset = R_offset + spline_meta_.NumParameters();

    SO3T R_IkToG;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    Vec3T p_IkinG;
    CeresSplineHelperJet<T, _N>::template evaluate<3, 0>(sKnots + P_offset, u,
                                                         inv_dt_, &p_IkinG);

    Eigen::Map<SO3T const> const R_LtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_LinI(sKnots[Kont_offset + 1]);

    SO3T R_LkToG = R_IkToG * R_LtoI;
    Vec3T p_LkinG = R_IkToG * p_LinI + p_IkinG;

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_LkToG * pose_data_.orientation.inverse()).log();

    residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_LkinG - pose_data_.position);
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
class LiDAROrientationFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LiDAROrientationFactor(const PoseData& pose_data,
                         const SplineMeta<_N>& spline_meta, double rot_weight)
      : pose_data_(pose_data),
        spline_meta_(spline_meta),
        rot_weight_(rot_weight) {
    inv_dt_ = 1.0 / spline_meta_.segments.begin()->dt;
  }

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    T t = T(pose_data_.timestamp);

    size_t R_offset;  // should be zero if not estimate time offset
    T u;
    spline_meta_.ComputeSplineIndex(t, R_offset, u);

    SO3T R_IkToG;
    CeresSplineHelperJet<T, _N>::template evaluate_lie<Sophus::SO3>(
        sKnots + R_offset, u, inv_dt_, &R_IkToG);

    int Kont_offset = spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_LtoI(sKnots[Kont_offset]);

    SO3T R_LkToG = R_IkToG * R_LtoI;

    Eigen::Map<Tangent> residuals(sResiduals);
    residuals =
        T(rot_weight_) * (R_LkToG * pose_data_.orientation.inverse()).log();

    return true;
  }

 private:
  PoseData pose_data_;
  SplineMeta<_N> spline_meta_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class SO3KnotFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SO3KnotFactor(const Sophus::SO3d& so3_knots, double so3_knot_weight)
      : so3_knots_(so3_knots), so3_knot_weight_(so3_knot_weight) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    Eigen::Map<SO3T const> const knot(sKnots[0]);
    Eigen::Map<SO3T const> const R_LtoI(sKnots[1]);

    Eigen::Map<Tangent> residuals(sResiduals);
    residuals =
        T(so3_knot_weight_) * ((knot * R_LtoI * so3_knots_.inverse()).log());

    return true;
  }

 private:
  Sophus::SO3d so3_knots_;
  double so3_knot_weight_;
};

}  // namespace clins

#endif
