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

#ifndef VOXEL_FILTER_H
#define VOXEL_FILTER_H

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_data/lidar_data.h>

#include <utils/tic_toc.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

namespace clins {

struct cloud_point_index_idx {
  unsigned int idx;
  unsigned int cloud_point_index;

  cloud_point_index_idx() = default;
  cloud_point_index_idx(unsigned int idx_, unsigned int cloud_point_index_)
      : idx(idx_), cloud_point_index(cloud_point_index_) {}
  bool operator<(const cloud_point_index_idx &p) const { return (idx < p.idx); }
};

template <typename PointType>
class VoxelFilter {
 public:
  VoxelFilter() {}

  void SetResolution(float resolution) {
    resolution_ = resolution;
    inverse_resolution_ = 1.0 / resolution_;
  }

  void SetInputCloud(
      const typename pcl::PointCloud<PointType>::Ptr input_cloud) {
    input_cloud_ = input_cloud;
  }

  float PointDistanceSquare(Eigen::Vector3f p1, Eigen::Vector3f p2) {
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) +
           (p1[1] - p2[1]) * (p1[1] - p2[1]) +
           (p1[2] - p2[2]) * (p1[2] - p2[2]);
  }

  void Filter(typename pcl::PointCloud<PointType>::Ptr output_cloud) {
    /// 栅格化
    pcl::getMinMax3D<PointType>(*input_cloud_, min_map_, max_map_);
    min_b_[0] = static_cast<int>(floor(min_map_[0] * inverse_resolution_));
    max_b_[0] = static_cast<int>(floor(max_map_[0] * inverse_resolution_));
    min_b_[1] = static_cast<int>(floor(min_map_[1] * inverse_resolution_));
    max_b_[1] = static_cast<int>(floor(max_map_[1] * inverse_resolution_));
    min_b_[2] = static_cast<int>(floor(min_map_[2] * inverse_resolution_));
    max_b_[2] = static_cast<int>(floor(max_map_[2] * inverse_resolution_));

    div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
    div_b_[3] = 0;
    divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

    std::vector<cloud_point_index_idx> index_vector;
    index_vector.reserve(input_cloud_->points.size());

    for (unsigned int i = 0; i < input_cloud_->points.size(); i++) {
      PointType p = input_cloud_->points[i];
      if (!pcl_isfinite(p.x) || !pcl_isfinite(p.y) || !pcl_isfinite(p.z))
        continue;

      int ijk0 = static_cast<int>(floor(p.x * inverse_resolution_) -
                                  static_cast<float>(min_b_[0]));
      int ijk1 = static_cast<int>(floor(p.y * inverse_resolution_) -
                                  static_cast<float>(min_b_[1]));
      int ijk2 = static_cast<int>(floor(p.z * inverse_resolution_) -
                                  static_cast<float>(min_b_[2]));

      int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
      index_vector.emplace_back(static_cast<unsigned int>(idx), i);
    }

    auto rightshift_func = [](const cloud_point_index_idx &x,
                              const unsigned offset) {
      return x.idx >> offset;
    };
    boost::sort::spreadsort::integer_sort(index_vector.begin(),
                                          index_vector.end(), rightshift_func);

    unsigned int total = 0;
    unsigned int index = 0;

    std::vector<std::pair<unsigned int, unsigned int> >
        first_and_last_indices_vector;
    first_and_last_indices_vector.reserve(index_vector.size());
    while (index < index_vector.size()) {
      unsigned int i = index + 1;
      while (i < index_vector.size() &&
             index_vector[i].idx == index_vector[index].idx)
        ++i;
      if (i - index >= 0) {
        ++total;
        first_and_last_indices_vector.emplace_back(index, i);
      }
      index = i;
    }

    for (auto leaf : first_and_last_indices_vector) {
      Eigen::Vector3f centroid(0, 0, 0);
      for (int i = leaf.first; i < leaf.second; i++) {
        centroid += Eigen::Vector3f(
            input_cloud_->points[index_vector[i].cloud_point_index].x,
            input_cloud_->points[index_vector[i].cloud_point_index].y,
            input_cloud_->points[index_vector[i].cloud_point_index].z);
      }
      centroid /= static_cast<float>(leaf.second - leaf.first);
      PointType p;
      float dis = 10000;
      for (int i = leaf.first; i < leaf.second; i++) {
        Eigen::Vector3f cp = Eigen::Vector3f(
            input_cloud_->points[index_vector[i].cloud_point_index].x,
            input_cloud_->points[index_vector[i].cloud_point_index].y,
            input_cloud_->points[index_vector[i].cloud_point_index].z);
        float disSqu = PointDistanceSquare(cp, centroid);
        if (disSqu <= dis) {
          p = input_cloud_->points[index_vector[i].cloud_point_index];
          dis = disSqu;
        }
      }

      output_cloud->push_back(p);
    }
  }

 private:
  float resolution_;
  float inverse_resolution_;

  typename pcl::PointCloud<PointType>::Ptr input_cloud_;

  Eigen::Vector4f min_map_;
  Eigen::Vector4f max_map_;

  Eigen::Vector4i min_b_, max_b_, div_b_, divb_mul_;
};

}  // namespace clins

#endif
