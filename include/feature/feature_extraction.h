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

#ifndef FEATURE_EXTRATION_H
#define FEATURE_EXTRATION_H

#include <clins/feature_cloud.h>
#include <feature/voxel_filter.h>
#include <ros/ros.h>
#include <sensor_data/lidar_data.h>
#include <sensor_msgs/PointCloud2.h>
#include <utils/tic_toc.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_data/calibration.hpp>

namespace clins {

struct smoothness_t {
  float value;
  size_t ind;
};

struct by_value {
  bool operator()(smoothness_t const& left, smoothness_t const& right) {
    return left.value < right.value;
  }
};

class FeatureExtraction {
 public:
  FeatureExtraction(const YAML::Node& node);

  void LidarHandler(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg);

  void LidarHandler(RTPointCloud::Ptr raw_cloud,
                    RTPointCloud::Ptr undistort_cloud);

  void AllocateMemory();

  void ResetParameters();

  bool CachePointCloud(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg);

  /// VLP_16 time = dsr * 2.304 * 1e-6 + firing * 55.296 * 1e-6;
  void ProjectOrganizedCloud(const VPointCloud::Ptr cur_cloud,
                             cv::Mat& dist_image,
                             RTPointCloud::Ptr corresponding_cloud);

  /// project cloud to range mat
  void ProjectPointCloud(const RTPointCloud::Ptr cur_cloud, cv::Mat& dist_image,
                         RTPointCloud::Ptr corresponding_cloud);

  void ProjectPointCloud(const RTPointCloud::Ptr cur_cloud,
                         const RTPointCloud::Ptr undistort_cloud,
                         cv::Mat& dist_image,
                         RTPointCloud::Ptr cur_corresponding_cloud,
                         RTPointCloud::Ptr undistort_correspondig_cloud);

  void CloudExtraction();

  void CaculateSmoothness();

  void MarkOccludedPoints();

  void ExtractFeatures();

  void PublishCloud(std::string frame_id);

  void RTPointCloudVoxelFilter(RTPointCloud::Ptr input_cloud,
                               RTPointCloud::Ptr output_cloud);

  template <typename TPoint>
  float PointDistance(const TPoint& p) const;

  inline RTPointCloud::Ptr get_corner_features() const {
    return p_corner_cloud;
  }

  inline RTPointCloud::Ptr get_raw_corner_features() const {
    return p_raw_corner_cloud;
  }

  inline RTPointCloud::Ptr get_surface_features() const {
    return p_surface_cloud;
  }

  inline RTPointCloud::Ptr get_raw_surface_features() const {
    return p_raw_surface_cloud;
  }

  bool CheckMsgFields(const sensor_msgs::PointCloud2& cloud_msg,
                      std::string fields_name = "ring");

 private:
  ros::NodeHandle nh;

  ros::Subscriber sub_lidar;
  ros::Publisher pub_corner_cloud;
  ros::Publisher pub_surface_cloud;
  ros::Publisher pub_full_cloud;
  ros::Publisher pub_feature_cloud;

  sensor_msgs::PointCloud2 cur_cloud_msg;

  double min_distance_;
  double max_distance_;

  int n_scan;
  int horizon_scan;
  cv::Mat range_mat;
  RTPointCloud::Ptr p_full_cloud;
  RTPointCloud::Ptr p_raw_cloud;

  RTPointCloud::Ptr p_extracted_cloud;

  std::string lidar_topic;

  std::vector<float> point_range_list;
  std::vector<int> point_column_id;
  std::vector<int> start_ring_index;
  std::vector<int> end_ring_index;

  std::vector<smoothness_t> cloud_smoothness;
  float* cloud_curvature;
  int* cloud_neighbor_picked;
  int* cloud_label;

  /// feature
  RTPointCloud::Ptr p_corner_cloud;
  RTPointCloud::Ptr p_surface_cloud;

  RTPointCloud::Ptr p_raw_corner_cloud;
  RTPointCloud::Ptr p_raw_surface_cloud;

  /// LOAM feature threshold
  float edge_threshold;
  float surf_threshold;

  VoxelFilter<RTPoint> down_size_filter;
  float odometry_surface_leaf_size;

  bool undistort_scan_before_extraction_;

  bool use_corner_feature_;
};

}  // namespace clins

#endif
