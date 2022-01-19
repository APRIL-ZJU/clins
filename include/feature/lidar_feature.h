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

#ifndef LIDAR_FEATURE_H
#define LIDAR_FEATURE_H

#include <ceres/ceres.h>
#include <sensor_data/lidar_data.h>
#include <Eigen/Eigen>

namespace clins {

struct LiDARFeature {
  LiDARFeature()
      : timestamp(0),
        time_max(0),
        corner_features(new PosCloud),
        surface_features(new PosCloud),
        full_features(new PosCloud) {}

  LiDARFeature(const LiDARFeature& fea) {
    timestamp = fea.timestamp;
    time_max = fea.time_max;
    corner_features = PosCloud::Ptr(new PosCloud);
    surface_features = PosCloud::Ptr(new PosCloud);
    full_features = PosCloud::Ptr(new PosCloud);
    *corner_features = *fea.corner_features;
    *surface_features = *fea.surface_features;
    *full_features = *fea.full_features;
  }

  LiDARFeature& operator=(const LiDARFeature& fea) {
    if (this != &fea) {
      LiDARFeature temp(fea);
      this->timestamp = temp.timestamp;
      this->time_max = temp.time_max;

      PosCloud::Ptr p_temp = temp.corner_features;
      temp.corner_features = this->corner_features;
      this->corner_features = p_temp;

      p_temp = temp.surface_features;
      temp.surface_features = this->surface_features;
      this->surface_features = p_temp;

      p_temp = temp.full_features;
      temp.full_features = this->full_features;
      this->full_features = p_temp;
    }

    return *this;
  }

  void Clear() {
    timestamp = 0;
    time_max = 0;
    corner_features->clear();
    surface_features->clear();
    full_features->clear();
  }

  double timestamp;
  double time_max;  // [timestamp, max_time] of full_features
  PosCloud::Ptr corner_features;
  PosCloud::Ptr surface_features;
  PosCloud::Ptr full_features;
};

enum GeometryType { Line = 0, Plane };

struct PointCorrespondence {
  double t_point;
  double t_map;
  Eigen::Vector3d point;
  Eigen::Vector3d point_raw;

  GeometryType geo_type;
  Eigen::Vector4d geo_plane;

  Eigen::Vector3d geo_normal;
  Eigen::Vector3d geo_point;
};
}  // namespace clins

#endif
