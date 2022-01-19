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

#ifndef VICON_FACTOR_H
#define VICON_FACTOR_H
#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/spline/ceres_spline_helper_jet.h>
#include <basalt/spline/spline_segment.h>
#include <ceres/ceres.h>
#include <sensor_data/imu_data.h>
#include <sophus/so3.hpp>

namespace clins {
using namespace basalt;

template <int _N>
class ViconPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ViconPoseFactor(const PoseData& vicon_data, const SplineMeta<_N>& spline_meta,
                  double pos_weight, double rot_weight)
      : vicon_data_(vicon_data),
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

    T t_offset = sKnots[Kont_offset + 4][0];
    T t = T(vicon_data_.timestamp) + t_offset;

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

    Eigen::Map<SO3T const> const R_VtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_VinI(sKnots[Kont_offset + 1]);
    Eigen::Map<SO3T const> const R_ViconCoor(sKnots[Kont_offset + 2]);
    Eigen::Map<Vec3T const> const p_ViconCoor(sKnots[Kont_offset + 3]);

    SO3T R_VkToG = R_IkToG * R_VtoI;
    Vec3T p_VkinG = R_IkToG * p_VinI + p_IkinG;

    SO3T R_ViconToG = R_ViconCoor * vicon_data_.orientation.cast<T>();
    Vec3T p_ViconinG =
        R_ViconCoor * vicon_data_.position.cast<T>() + p_ViconCoor;

    Eigen::Map<Vec6T> residuals(sResiduals);
    residuals.template block<3, 1>(0, 0) =
        T(rot_weight_) * (R_VkToG * R_ViconToG.inverse()).log();

    residuals.template block<3, 1>(3, 0) =
        T(pos_weight_) * (p_VkinG - p_ViconinG);
    return true;
  }

 private:
  PoseData vicon_data_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double rot_weight_;
  double inv_dt_;
};

template <int _N>
class GPSPoseFactor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GPSPoseFactor(const PoseData& GPS_data, const SplineMeta<_N>& spline_meta,
                double pos_weight)
      : GPS_data_(GPS_data),
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

    T t = T(GPS_data_.timestamp);
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

    size_t Kont_offset = 2 * spline_meta_.NumParameters();
    Eigen::Map<SO3T const> const R_LtoI(sKnots[Kont_offset]);
    Eigen::Map<Vec3T const> const p_LinI(sKnots[Kont_offset + 1]);
    Eigen::Map<SO3T const> const R_UTMCoor(sKnots[Kont_offset + 2]);
    Eigen::Map<Vec3T const> const p_UTMCoor(sKnots[Kont_offset + 3]);

    /// 用雷达的位姿去拟合
    Vec3T p_LkinG = R_IkToG * p_LinI + p_IkinG;

    Vec3T p_GPSinG = R_UTMCoor * GPS_data_.position.cast<T>() + p_UTMCoor;

    Eigen::Map<Vec3T> residuals(sResiduals);

    residuals.template block<3, 1>(0, 0) =
        T(pos_weight_) * (p_LkinG - p_GPSinG);
    return true;
  }

 private:
  PoseData GPS_data_;
  SplineMeta<_N> spline_meta_;
  double pos_weight_;
  double inv_dt_;
};

}  // namespace clins

#endif
