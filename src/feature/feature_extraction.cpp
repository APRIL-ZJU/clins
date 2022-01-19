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

#include <feature/feature_extraction.h>
#include <rosbag/bag.h>

using namespace std;

namespace clins {

FeatureExtraction::FeatureExtraction(const YAML::Node& node) {
  lidar_topic = node["lidar_topic"].as<std::string>();
  n_scan = node["N_SCAN"].as<int>();
  horizon_scan = node["Horizon_SCAN"].as<int>();

  edge_threshold = node["edge_threshold"].as<float>();
  surf_threshold = node["surf_threshold"].as<float>();
  odometry_surface_leaf_size = node["odometry_surface_leaf_size"].as<float>();

  undistort_scan_before_extraction_ =
      node["undistort_scan_before_extraction"].as<bool>();
  min_distance_ = node["min_distance"].as<double>();
  max_distance_ = node["max_distance"].as<double>();

  use_corner_feature_ = node["use_corner_feature"].as<bool>();

  AllocateMemory();
  ResetParameters();
}

void FeatureExtraction::LidarHandler(
    const sensor_msgs::PointCloud2::ConstPtr& lidar_msg) {
  if (!CachePointCloud(lidar_msg)) return;
  CloudExtraction();
  CaculateSmoothness();
  MarkOccludedPoints();
  ExtractFeatures();
  PublishCloud("map");
  ResetParameters();
}

void FeatureExtraction::LidarHandler(RTPointCloud::Ptr raw_cloud,
                                     RTPointCloud::Ptr undistort_cloud) {
  if (undistort_scan_before_extraction_)
    ProjectPointCloud(raw_cloud, undistort_cloud, range_mat, p_raw_cloud,
                      p_full_cloud);
  else
    ProjectPointCloud(raw_cloud, range_mat, p_full_cloud);

  CloudExtraction();
  CaculateSmoothness();
  MarkOccludedPoints();
  ExtractFeatures();
  ResetParameters();
}

void FeatureExtraction::AllocateMemory() {
  p_full_cloud.reset(new RTPointCloud());
  p_full_cloud->resize(n_scan * horizon_scan);
  p_raw_cloud.reset(new RTPointCloud());
  p_raw_cloud->resize(n_scan * horizon_scan);

  range_mat = cv::Mat(n_scan, horizon_scan, CV_32F, cv::Scalar::all(FLT_MAX));

  p_extracted_cloud.reset(new RTPointCloud());

  p_corner_cloud.reset(new RTPointCloud());
  p_surface_cloud.reset(new RTPointCloud());

  p_raw_corner_cloud.reset(new RTPointCloud());
  p_raw_surface_cloud.reset(new RTPointCloud());

  point_range_list.assign(n_scan * horizon_scan, 0);
  point_column_id.assign(n_scan * horizon_scan, 0);
  start_ring_index.assign(n_scan, 0);
  end_ring_index.assign(n_scan, 0);

  cloud_smoothness.resize(n_scan * horizon_scan);

  cloud_curvature = new float[n_scan * horizon_scan];
  cloud_neighbor_picked = new int[n_scan * horizon_scan];
  cloud_label = new int[n_scan * horizon_scan];

  down_size_filter.SetResolution(odometry_surface_leaf_size);
}

void FeatureExtraction::ResetParameters() {
  p_full_cloud->clear();
  p_full_cloud->resize(n_scan * horizon_scan);
  p_raw_cloud->clear();
  p_raw_cloud->resize(n_scan * horizon_scan);
  p_extracted_cloud->clear();
  range_mat = cv::Mat(n_scan, horizon_scan, CV_32F, cv::Scalar::all(FLT_MAX));
}

bool FeatureExtraction::CheckMsgFields(
    const sensor_msgs::PointCloud2& cloud_msg, std::string fields_name) {
  bool flag = false;
  for (int i = 0; i < cloud_msg.fields.size(); ++i) {
    if (cloud_msg.fields[i].name == fields_name) {
      flag = true;
    }
    if (flag) break;
  }
  return flag;

  if (!flag) {
    if (fields_name == "ring")
      cout << RED << "Point cloud ring channel not available, "
           << "please configure your point cloud data!" << RESET << endl;
    if (fields_name == "time")
      cout << RED << "Point cloud timestamp not available, "
           << "deskew function disabled, system will drift significantly!"
           << RESET << endl;
    ros::shutdown();
  }
}

bool FeatureExtraction::CachePointCloud(
    const sensor_msgs::PointCloud2::ConstPtr& lidar_msg) {
  cur_cloud_msg = *lidar_msg;

  static bool hasTimeField = false;

  /// Check ring channel and point time for the first msg
  static bool check_field = true;
  if (check_field) {
    bool ret = CheckMsgFields(cur_cloud_msg, "ring");
    ret = ret && CheckMsgFields(cur_cloud_msg, "time");
    check_field = false;

    if (ret) hasTimeField = true;
  }

  /// convert cloud
  if (hasTimeField) {
    RTPointCloud::Ptr cur_cloud(new RTPointCloud());
    pcl::fromROSMsg(cur_cloud_msg, *cur_cloud);

    ProjectPointCloud(cur_cloud, range_mat, p_full_cloud);
  } else {
    VPointCloud::Ptr cur_cloud(new VPointCloud());
    pcl::fromROSMsg(cur_cloud_msg, *cur_cloud);

    if (cur_cloud->isOrganized()) {
      ProjectOrganizedCloud(cur_cloud, range_mat, p_full_cloud);
    } else
      return false;
  }

  return true;
}

void FeatureExtraction::ProjectOrganizedCloud(
    const VPointCloud::Ptr cur_cloud, cv::Mat& dist_image,
    RTPointCloud::Ptr corresponding_cloud) {
  assert(cur_cloud->isOrganized() && cur_cloud->height == 16 &&
         cur_cloud->width ==
             1824 "[ProjectOrganizedCloud] input cloud should be organized");

  for (size_t column_id = 0; column_id < cur_cloud->width; ++column_id) {
    for (size_t row_id = 0; row_id < cur_cloud->height; ++row_id) {
      VPoint& p = cur_cloud->at(column_id, row_id);
      if (!pcl_isfinite(p.x) || !pcl_isfinite(p.y) || !pcl_isfinite(p.z))
        continue;

      if (row_id < 0 || row_id >= 16) continue;
      if (column_id < 0 || column_id >= horizon_scan) continue;

      float range = PointDistance<VPoint>(p);
      if (range < min_distance_ || range > max_distance_) continue;
      if (dist_image.at<float>(row_id, column_id) != FLT_MAX) continue;
      dist_image.at<float>(row_id, column_id) = range;

      /// 注意这个 index 与 cur_cloud 的索引不一致
      int index = column_id + row_id * horizon_scan;

      RTPoint rtpoint;
      rtpoint.x = p.x;
      rtpoint.y = p.y;
      rtpoint.z = p.z;
      rtpoint.intensity = p.intensity;
      rtpoint.ring = row_id;
      rtpoint.time = row_id * 2.304 * 1e-6 + column_id * 55.296 * 1e-6;
      corresponding_cloud->points[index] = rtpoint;
    }
  }
}

void FeatureExtraction::ProjectPointCloud(
    const RTPointCloud::Ptr cur_cloud, cv::Mat& dist_image,
    RTPointCloud::Ptr corresponding_cloud) {
  for (const RTPoint& p : cur_cloud->points) {
    if (!pcl_isfinite(p.x) || !pcl_isfinite(p.y) || !pcl_isfinite(p.z))
      continue;
    int row_id = p.ring;
    if (row_id < 0 || row_id >= n_scan) continue;
    float horizon_angle = atan2(p.x, p.y) * 180.0 / M_PI;

    float angle_resolution = 360.0 / float(horizon_scan);
    int column_id =
        -round((horizon_angle - 90.0) / angle_resolution) + horizon_scan / 2;
    if (column_id >= horizon_scan) column_id -= horizon_scan;
    if (column_id < 0 || column_id >= horizon_scan) continue;

    float range = PointDistance<RTPoint>(p);
    if (range < min_distance_ || range > max_distance_) continue;

    if (dist_image.at<float>(row_id, column_id) != FLT_MAX) continue;
    dist_image.at<float>(row_id, column_id) = range;

    int index = column_id + row_id * horizon_scan;
    corresponding_cloud->points[index] = p;
  }
}

void FeatureExtraction::ProjectPointCloud(
    const RTPointCloud::Ptr cur_cloud, const RTPointCloud::Ptr undistort_cloud,
    cv::Mat& dist_image, RTPointCloud::Ptr cur_corresponding_cloud,
    RTPointCloud::Ptr undistort_correspondig_cloud) {
  for (size_t i = 0; i < undistort_cloud->points.size(); i++) {
    RTPoint p = undistort_cloud->points[i];
    if (!pcl_isfinite(p.x) || !pcl_isfinite(p.y) || !pcl_isfinite(p.z))
      continue;
    int row_id = p.ring;
    if (row_id < 0 || row_id >= n_scan) continue;
    float horizon_angle = atan2(p.x, p.y) * 180.0 / M_PI;

    float angle_resolution = 360.0 / float(horizon_scan);
    int column_id =
        -round((horizon_angle - 90.0) / angle_resolution) + horizon_scan / 2;
    if (column_id >= horizon_scan) column_id -= horizon_scan;
    if (column_id < 0 || column_id >= horizon_scan) continue;

    float range = PointDistance<RTPoint>(p);
    if (range < min_distance_ || range > max_distance_) continue;

    if (dist_image.at<float>(row_id, column_id) != FLT_MAX) continue;
    dist_image.at<float>(row_id, column_id) = range;

    int index = column_id + row_id * horizon_scan;
    p.intensity = index;
    cur_corresponding_cloud->points[index] = cur_cloud->points[i];
    undistort_correspondig_cloud->points[index] = p;
  }
}

void FeatureExtraction::CloudExtraction() {
  int point_index = 0;
  for (int i = 0; i < n_scan; i++) {
    start_ring_index[i] = point_index - 1 + 5;
    for (int j = 0; j < horizon_scan; j++) {
      if (range_mat.at<float>(i, j) != FLT_MAX) {
        point_column_id[point_index] = j;
        point_range_list[point_index] = range_mat.at<float>(i, j);
        p_extracted_cloud->push_back(
            p_full_cloud->points[j + i * horizon_scan]);
        point_index++;
      }
    }
    end_ring_index[i] = point_index - 1 - 5;
  }
}

void FeatureExtraction::CaculateSmoothness() {
  for (int i = 5; i < p_extracted_cloud->points.size() - 5; i++) {
    float diff_range = point_range_list[i - 5] + point_range_list[i - 4] +
                       point_range_list[i - 3] + point_range_list[i - 2] +
                       point_range_list[i - 1] - point_range_list[i] * 10 +
                       point_range_list[i + 1] + point_range_list[i + 2] +
                       point_range_list[i + 3] + point_range_list[i + 4] +
                       point_range_list[i + 5];
    cloud_curvature[i] = diff_range * diff_range;
    cloud_neighbor_picked[i] = 0;
    cloud_label[i] = 0;
    cloud_smoothness[i].value = cloud_curvature[i];
    cloud_smoothness[i].ind = i;
  }
}

void FeatureExtraction::MarkOccludedPoints() {
  for (int i = 5; i < p_extracted_cloud->points.size() - 6; i++) {
    float depth1 = point_range_list[i];
    float depth2 = point_range_list[i + 1];
    int column_diff =
        std::abs(int(point_column_id[i + 1] - point_column_id[i]));

    if (column_diff < 10) {
      if (depth1 - depth2 > 0.3) {
        cloud_neighbor_picked[i - 5] = 1;
        cloud_neighbor_picked[i - 4] = 1;
        cloud_neighbor_picked[i - 3] = 1;
        cloud_neighbor_picked[i - 2] = 1;
        cloud_neighbor_picked[i - 1] = 1;
        cloud_neighbor_picked[i] = 1;
      } else if (depth2 - depth1 > 0.3) {
        cloud_neighbor_picked[i + 1] = 1;
        cloud_neighbor_picked[i + 2] = 1;
        cloud_neighbor_picked[i + 3] = 1;
        cloud_neighbor_picked[i + 4] = 1;
        cloud_neighbor_picked[i + 5] = 1;
        cloud_neighbor_picked[i + 6] = 1;
      }
    }

    float diff1 =
        std::abs(float(point_range_list[i - 1] - point_range_list[i]));
    float diff2 =
        std::abs(float(point_range_list[i + 1] - point_range_list[i]));

    if (diff1 > 0.02 * point_range_list[i] &&
        diff2 > 0.02 * point_range_list[i])
      cloud_neighbor_picked[i] = 1;
  }
}

void FeatureExtraction::ExtractFeatures() {
  p_corner_cloud->clear();
  p_surface_cloud->clear();

  RTPointCloud::Ptr surface_cloud_scan(new RTPointCloud());
  RTPointCloud::Ptr surface_cloud_scan_downsample(new RTPointCloud());

  for (int i = 0; i < n_scan; i++) {
    surface_cloud_scan->clear();

    for (int j = 0; j < 6; j++) {
      int sp = (start_ring_index[i] * (6 - j) + end_ring_index[i] * j) / 6;
      int ep =
          (start_ring_index[i] * (5 - j) + end_ring_index[i] * (j + 1)) / 6 - 1;
      if (sp >= ep) continue;
      std::sort(cloud_smoothness.begin() + sp, cloud_smoothness.begin() + ep,
                by_value());

      int largest_picked_num = 0;
      for (int k = ep; k >= sp; k--) {
        int index = cloud_smoothness[k].ind;
        if (cloud_neighbor_picked[index] == 0 &&
            cloud_curvature[index] > edge_threshold) {
          largest_picked_num++;
          if (largest_picked_num <= 20) {
            cloud_label[index] = 1;
            p_corner_cloud->push_back(p_extracted_cloud->points[index]);
          } else {
            break;
          }

          cloud_neighbor_picked[index] = 1;
          for (int l = 1; l <= 5; l++) {
            int column_diff = std::abs(int(point_column_id[index + l] -
                                           point_column_id[index + l - 1]));
            if (column_diff > 10) break;
            cloud_neighbor_picked[index + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int column_diff = std::abs(int(point_column_id[index + l] -
                                           point_column_id[index + l + 1]));
            if (column_diff > 10) break;
            cloud_neighbor_picked[index + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        int index = cloud_smoothness[k].ind;
        if (cloud_neighbor_picked[index] == 0 &&
            cloud_curvature[index] < surf_threshold) {
          cloud_label[index] = -1;
          cloud_neighbor_picked[index] = 1;

          for (int l = 1; l <= 5; l++) {
            int column_diff = std::abs(int(point_column_id[index + l] -
                                           point_column_id[index + l - 1]));
            if (column_diff > 10) break;
            cloud_neighbor_picked[index + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int column_diff = std::abs(int(point_column_id[index + l] -
                                           point_column_id[index + l + 1]));
            if (column_diff > 10) break;
            cloud_neighbor_picked[index + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        if (cloud_label[k] <= 0) {
          surface_cloud_scan->push_back(p_extracted_cloud->points[k]);
        }
      }
    }

    surface_cloud_scan_downsample->clear();
    down_size_filter.SetInputCloud(surface_cloud_scan);
    down_size_filter.Filter(surface_cloud_scan_downsample);
    *p_surface_cloud += *surface_cloud_scan_downsample;

    if (undistort_scan_before_extraction_) {
      p_raw_corner_cloud->clear();
      p_raw_surface_cloud->clear();
      if (use_corner_feature_) {
        p_raw_corner_cloud->resize(p_corner_cloud->points.size());
        for (size_t i = 0; i < p_corner_cloud->points.size(); i++) {
          int index = int(p_corner_cloud->points[i].intensity);
          p_raw_corner_cloud->points[i] = p_raw_cloud->points[index];
        }
      }

      p_raw_surface_cloud->resize(p_surface_cloud->points.size());
      for (size_t i = 0; i < p_surface_cloud->points.size(); i++) {
        int index = int(p_surface_cloud->points[i].intensity);
        p_raw_surface_cloud->points[i] = p_raw_cloud->points[index];
      }
    } else {
      p_raw_corner_cloud = p_corner_cloud;
      p_raw_surface_cloud = p_surface_cloud;
    }
  }
}

void FeatureExtraction::PublishCloud(std::string frame_id) {
  sensor_msgs::PointCloud2 corner_msg;
  sensor_msgs::PointCloud2 surface_msg;
  sensor_msgs::PointCloud2 full_msg;
  clins::feature_cloud feature_msg;

  pcl::toROSMsg(*p_corner_cloud, corner_msg);
  pcl::toROSMsg(*p_surface_cloud, surface_msg);
  pcl::toROSMsg(*p_extracted_cloud, full_msg);

  corner_msg.header.stamp = cur_cloud_msg.header.stamp;
  corner_msg.header.frame_id = frame_id;
  surface_msg.header.stamp = cur_cloud_msg.header.stamp;
  surface_msg.header.frame_id = frame_id;
  full_msg.header.stamp = cur_cloud_msg.header.stamp;
  full_msg.header.frame_id = frame_id;

  feature_msg.header = cur_cloud_msg.header;
  feature_msg.full_cloud = full_msg;
  feature_msg.corner_cloud = corner_msg;
  feature_msg.surface_cloud = surface_msg;

  pub_feature_cloud.publish(feature_msg);

  if (pub_corner_cloud.getNumSubscribers() != 0)
    pub_corner_cloud.publish(corner_msg);
  if (pub_surface_cloud.getNumSubscribers() != 0)
    pub_surface_cloud.publish(surface_msg);
  if (pub_full_cloud.getNumSubscribers() != 0) pub_full_cloud.publish(full_msg);

  //    rosbag::Bag bagWrite;
  //    bagWrite.open("/home/ha/rosbag/liso-bag/simu_bag/sim_feature.bag",
  //    rosbag::bagmode::Append); bagWrite.write("/feature_cloud",
  //    feature_msg.header.stamp, feature_msg); bagWrite.close();
}

void FeatureExtraction::RTPointCloudVoxelFilter(
    RTPointCloud::Ptr input_cloud, RTPointCloud::Ptr output_cloud) {}

template <typename TPoint>
float FeatureExtraction::PointDistance(const TPoint& p) const {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

}  // namespace clins
