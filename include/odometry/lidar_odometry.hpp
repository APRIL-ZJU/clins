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

#ifndef LIDAR_ODOMETRY_HPP
#define LIDAR_ODOMETRY_HPP

#include <feature/feature_extraction.h>
#include <sensor_data/imu_data.h>
#include <basalt/utils/eigen_utils.hpp>
#include <memory>
#include <trajectory/se3_trajectory.hpp>
#include <trajectory/trajectory_manager.hpp>

// allows us to use pcl::transformPointCloud function
#include <pcl/common/transforms.h>

#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Path.h>
#include <pcl/registration/icp.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <utils/tic_toc.h>
#include <visualization_msgs/MarkerArray.h>
#include <trajectory/trajectory_viewer.hpp>

#include <mutex>
#include <unordered_map>

#include <ceres/solver.h>
#include <pose_graph/pose_graph_3d_error_term.h>
#include <pose_graph/types.h>

namespace clins {

template <int _N>
class LidarOdometry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<LidarOdometry<_N>> Ptr;

  LidarOdometry(const YAML::Node& node, typename Trajectory<_N>::Ptr traj);

  void FeatureCloudHandler(double scan_timestamp, double scan_time_max,
                           const RTPointCloud::Ptr raw_corner_feature,
                           const RTPointCloud::Ptr raw_surface_feature,
                           const RTPointCloud::Ptr raw_full_feature,
                           const RTPointCloud::Ptr corner_feature = nullptr,
                           const RTPointCloud::Ptr surface_feature = nullptr,
                           const RTPointCloud::Ptr full_feature = nullptr);

  void EstimateIMUMeasurement(double scan_time = -1, double scan_max_time = -1);

  bool UpdateOdometry();

  void LoopClosureHandler();

  void SetTrajectoryManager(std::shared_ptr<TrajectoryManager<_N>> manager) {
    trajectory_manager_ = manager;
  }

  void SetCachePath(std::string cache_path) { cache_path_ = cache_path; }

  PosCloud::Ptr GetCloudKeyPose() { return cloud_key_pos_; }

  void SaveKeyFrameCloud(std::string path);

  std::map<int, int> GetHistoryLoopClosureInfo() { return history_loop_info_; }

  void CacheLoopICPResult(GPointCloud::Ptr target_cloud,
                          GPointCloud::Ptr source_cloud, Eigen::Matrix4f& guess,
                          Eigen::Matrix4f& icp_reuslt);

 protected:
  void ExtractSurroundFeatures(const double cur_time);

  void DownsampleCurrentScan(LiDARFeature& kf_in, LiDARFeature& kf_out);

  void UpdateKeyFrames();

  const bool isKeyFrame(double time) const;

  void PointCloudConvert(const RTPointCloud::Ptr input_cloud,
                         double scan_timestamp, PosCloud::Ptr output_cloud,
                         double* max_time = NULL);

  const double PointDistance(const PosPoint& p1, const PosPoint& p2) const {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
                (p1.z - p2.z) * (p1.z - p2.z));
  }

  /// target : feature_map_ds
  bool FindCorrespondence(const LiDARFeature& feature_cur,
                          const LiDARFeature& feature_cur_in_M);

  bool SetTargetFeature(const LiDARFeature& feature_map);

  bool ScanMatch(int iter_num = 0);

  /// remove timestamp field
  void PosCloudConvert(const PosCloud& input_cloud, VPointCloud& output_cloud);

  void DownSampleCorrespondence();

  bool DetectLoopClosure(int& latest_index, int& closest_index);

  void FindNearbyKeyFrames(GPointCloud::Ptr cloud, const int key,
                           const int search_num, double& min_time,
                           double& max_time);

  void FindNearbyKeyFrames(GPointCloud::Ptr cloud, const int key,
                           const int search_num, double& min_time,
                           double& max_time, Eigen::Matrix4d& pose_in_G);

  double ComputePoseHeightDiff(int target_index, int source_index);

  void ComputeLoopClosureParam(LoopClosureOptimizationParam& param);

  void DiscretedPoseGraph(LoopClosureOptimizationParam& param);

  bool PointToPointICP(GPointCloud::Ptr target_cloud,
                       GPointCloud::Ptr source_cloud,
                       double correspondence_distance, Eigen::Matrix4f guess,
                       Eigen::Matrix4f& transform, float& fitness_score);

  bool PointToPlaneICP(GPointCloud::Ptr target_cloud,
                       GPointCloud::Ptr source_cloud,
                       Eigen::Matrix4f& transform, float& fitness_score);

  /// position of keyframe
  PosCloud::Ptr cloud_key_pos_;
  PosCloud::Ptr cloud_key_pos_xy_;

  /// clone of keyframe's position
  PosCloud::Ptr cloud_key_pos_xy_copied_;

  std::shared_ptr<Trajectory<_N>> trajectory_;

  std::shared_ptr<TrajectoryManager<_N>> trajectory_manager_;

  LiDARFeature feature_cur_;
  LiDARFeature feature_cur_ds_;

  LiDARFeature feature_map_;
  LiDARFeature feature_map_ds_;

  /// All keyframes
  std::map<double, LiDARFeature> local_feature_container_;

  /// The latest frames, to check whether they are keyframes
  std::map<double, LiDARFeature> cache_feature_container_;

 private:
  VoxelFilter<PosPoint> surrounding_key_pos_voxel_filter_;
  VoxelFilter<PosPoint> corner_features_voxel_filter_;
  VoxelFilter<PosPoint> surface_features_voxel_filter_;

  pcl::VoxelGrid<GPoint> loop_closure_match_voxel_filter_;

  float keyframe_search_radius_;
  float keyframe_adding_angle_threshold_;
  float keyframe_adding_dist_meter_;

  int edge_min_valid_num_;
  int surf_min_valid_num_;

  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_corner_map_;
  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_surface_map_;

  pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_history_key_poses_;

  /// The correspondences od current scan
  Eigen::aligned_vector<PointCorrespondence> point_correspondence_;

  bool use_corner_feature_;

  bool update_key_frame_;

  /// Downsample parameter of correspondence
  int downsample_num_;

  /// Loop Closure
  bool loop_closure_use_full_cloud_;
  bool loop_closure_icp_recognition_manual_check_;
  double history_key_frame_search_radius_;
  double history_key_frame_time_diff_;
  int history_key_frame_index_diff_;
  int history_key_frame_search_num_;
  double history_key_frame_fitness_score_;
  double loop_closure_downsample_leaf_size_;
  double neighbor_edg_sample_time_;
  LoopClosureWeights lc_weights_;

  // loop closure
  std::map<int, int> history_loop_info_;
  std::map<int, RelativePoseData> history_loop_edgs_;
  std::mutex loop_mutex_;

  std::string cache_path_;
};

template <int _N>
LidarOdometry<_N>::LidarOdometry(const YAML::Node& node,
                                 typename Trajectory<_N>::Ptr traj)
    : cloud_key_pos_(new PosCloud),
      cloud_key_pos_xy_(new PosCloud),
      cloud_key_pos_xy_copied_(new PosCloud),
      trajectory_(traj),
      kdtree_corner_map_(new pcl::KdTreeFLANN<PosPoint>()),
      kdtree_surface_map_(new pcl::KdTreeFLANN<PosPoint>()),
      kdtree_history_key_poses_(new pcl::KdTreeFLANN<PosPoint>()),
      use_corner_feature_(true),
      update_key_frame_(false),
      downsample_num_(1) {
  float keyframe_density = node["keyframe_density"].as<float>();
  float corner_leaf_size = node["corner_leaf_size"].as<float>();
  float surface_leaf_size = node["surface_leaf_size"].as<float>();
  surrounding_key_pos_voxel_filter_.SetResolution(keyframe_density);
  corner_features_voxel_filter_.SetResolution(corner_leaf_size);
  surface_features_voxel_filter_.SetResolution(surface_leaf_size);

  keyframe_search_radius_ = node["keyframe_search_radius"].as<float>();
  keyframe_adding_angle_threshold_ =
      node["keyframe_adding_angle_threshold"].as<float>();
  keyframe_adding_dist_meter_ = node["keyframe_adding_dist_meter"].as<float>();

  edge_min_valid_num_ = node["edge_min_valid_num"].as<int>();
  surf_min_valid_num_ = node["surf_min_valid_num"].as<int>();

  if (node["use_corner_feature"]) {
    use_corner_feature_ = node["use_corner_feature"].as<bool>();
  }

  if (node["downsample_num"]) {
    downsample_num_ = node["downsample_num"].as<int>();
  }

  loop_closure_use_full_cloud_ = node["loop_closure_use_full_cloud"].as<bool>();
  loop_closure_icp_recognition_manual_check_ =
      node["loop_closure_icp_recognition_manual_check"].as<bool>();
  history_key_frame_search_radius_ =
      node["history_key_frame_search_radius"].as<double>();
  history_key_frame_time_diff_ =
      node["history_key_frame_time_diff"].as<double>();
  history_key_frame_index_diff_ =
      node["history_key_frame_index_diff"].as<int>();
  history_key_frame_search_num_ =
      node["history_key_frame_search_num"].as<int>();
  history_key_frame_fitness_score_ =
      node["history_key_frame_fitness_score"].as<double>();
  loop_closure_downsample_leaf_size_ =
      node["loop_closure_downsample_leaf_size"].as<double>();
  neighbor_edg_sample_time_ = node["neighbor_edg_sample_time"].as<double>();

  lc_weights_.velocity_weight =
      node["loop_closure_weights"]["velocity_weight"].as<double>();
  lc_weights_.gyro_weight =
      node["loop_closure_weights"]["gyro_weight"].as<double>();
  lc_weights_.pose_graph_edge_pos_weight =
      node["loop_closure_weights"]["pose_graph_edge_pos_weight"].as<double>();
  lc_weights_.pose_graph_edge_rot_weight =
      node["loop_closure_weights"]["pose_graph_edge_rot_weight"].as<double>();

  loop_closure_match_voxel_filter_.setLeafSize(
      loop_closure_downsample_leaf_size_, loop_closure_downsample_leaf_size_,
      loop_closure_downsample_leaf_size_);
}

template <int _N>
void LidarOdometry<_N>::FeatureCloudHandler(
    double scan_timestamp, double scan_time_max,
    const RTPointCloud::Ptr raw_corner_feature,
    const RTPointCloud::Ptr raw_surface_feature,
    const RTPointCloud::Ptr raw_full_feature,
    const RTPointCloud::Ptr corner_feature,
    const RTPointCloud::Ptr surface_feature,
    const RTPointCloud::Ptr full_feature) {
  feature_cur_.Clear();
  feature_cur_.timestamp = scan_timestamp;
  feature_cur_.time_max = scan_time_max;
  if (use_corner_feature_) {
    PointCloudConvert(raw_corner_feature, feature_cur_.timestamp,
                      feature_cur_.corner_features);
  }
  PointCloudConvert(raw_surface_feature, feature_cur_.timestamp,
                    feature_cur_.surface_features);
  PointCloudConvert(raw_full_feature, feature_cur_.timestamp,
                    feature_cur_.full_features);
}

template <int _N>
void LidarOdometry<_N>::EstimateIMUMeasurement(double scan_time,
                                               double scan_max_time) {
  if (scan_time < 0 || scan_max_time < 0) {
    trajectory_manager_->IntegrateIMUMeasurement(feature_cur_.timestamp,
                                                 feature_cur_.time_max);
  } else {
    trajectory_manager_->IntegrateIMUMeasurement(scan_time, scan_max_time);
  }
}

template <int _N>
bool LidarOdometry<_N>::UpdateOdometry() {
  if (cloud_key_pos_->points.empty()) {
    UpdateKeyFrames();
    return false;
  }

  if (update_key_frame_) {
    ExtractSurroundFeatures(feature_cur_.timestamp);
    update_key_frame_ = false;
  }

  DownsampleCurrentScan(feature_cur_, feature_cur_ds_);

  ScanMatch();

  TrajectoryViewer::PublishDenseCloud<4>(trajectory_, feature_map_ds_,
                                         feature_cur_ds_);

  UpdateKeyFrames();
}

template <int _N>
void LidarOdometry<_N>::LoopClosureHandler() {
  //  loop_mutex_.lock();
  *cloud_key_pos_xy_copied_ = *cloud_key_pos_xy_;
  if (cloud_key_pos_xy_copied_->points.size() == 0) return;
  //  loop_mutex_.unlock();

  int loop_cur_index = 0;
  int loop_history_index = 0;
  if (!DetectLoopClosure(loop_cur_index, loop_history_index)) {
    //    std::cout << RED << "DetectLoopClosure : " << loop_cur_index << " : "
    //              << loop_history_index << RESET << std::endl;
    return;
  }

  std::cout << "Loop Closure " << loop_cur_index << " <---> "
            << loop_history_index << std::endl;

  GPointCloud::Ptr cur_key_frame_cloud(new GPointCloud);
  GPointCloud::Ptr history_key_frame_cloud(new GPointCloud);
  double cur_key_frame_min_time;
  double cur_key_frame_max_time;
  double history_key_frame_min_time;
  double history_key_frame_max_time;
  Eigen::Matrix4d cur_key_frame_pose;
  Eigen::Matrix4d history_key_frame_pose;

  FindNearbyKeyFrames(cur_key_frame_cloud, loop_cur_index, 0,
                      cur_key_frame_min_time, cur_key_frame_max_time,
                      cur_key_frame_pose);
  FindNearbyKeyFrames(history_key_frame_cloud, loop_history_index,
                      history_key_frame_search_num_, history_key_frame_min_time,
                      history_key_frame_max_time, history_key_frame_pose);
  Eigen::Matrix4f guess_transform_matrix =
      (history_key_frame_pose.inverse() * cur_key_frame_pose).cast<float>();

  double delta_height =
      ComputePoseHeightDiff(loop_history_index, loop_cur_index);
  float fitness_score;
  Eigen::Matrix4f correction_transform;

  while (true) {
    bool icp_success = PointToPointICP(
        history_key_frame_cloud, cur_key_frame_cloud, delta_height * 2,
        guess_transform_matrix, correction_transform, fitness_score);

    if (!loop_closure_icp_recognition_manual_check_) {
      if (!icp_success) {
        std::cout << RED << " ICP Error " << fitness_score << RESET
                  << std::endl;
        return;
      } else {
        break;
      }
    }

    //  CacheLoopICPResult(history_key_frame_cloud, cur_key_frame_cloud,
    //                     guess_transform_matrix, correction_transform);

    TrajectoryViewer::PublishICPCloud(
        history_key_frame_cloud, cur_key_frame_cloud,
        history_key_frame_pose.cast<float>(), correction_transform);

    if (loop_closure_icp_recognition_manual_check_) {
      //      if (icp_success) break;
      std::cout << RED << " Is ICP registration successful ? (Y or N)" << RESET
                << std::endl;
      char con;

      std::cin >> con;
      if (con == 'y' || con == 'Y') {
        std::cout << GREEN << " ICP Success " << fitness_score << RESET
                  << std::endl;
        break;
      } else if (con == 'n' || con == 'N') {
        std::cout << "Please input dx , dy , dz : " << std::endl;
        float dx = 0, dy = 0, dz = 0;
        while (!(std::cin >> dx >> dy >> dz)) {
          std::cin.clear();
          std::cin.ignore();
          std::cin.sync();
          std::cout << "Input Error" << std::endl;
        }
        guess_transform_matrix = correction_transform;
        guess_transform_matrix(0, 3) += dx;
        guess_transform_matrix(1, 3) += dy;
        guess_transform_matrix(2, 3) += dz;

      } else if (con == 'r' || con == 'R') {
        return;
      } else {
        std::cout << "Input Error !" << std::endl;
      }
    }
  }

  LoopClosureOptimizationParam lco_param;
  lco_param.cur_key_frame_timestamp =
      cloud_key_pos_xy_copied_->points[loop_cur_index].timestamp;
  lco_param.cur_key_frame_max_time = cur_key_frame_max_time;
  lco_param.cur_key_frame_index = loop_cur_index;
  lco_param.history_key_frame_timestamp =
      cloud_key_pos_xy_copied_->points[loop_history_index].timestamp;
  lco_param.history_key_frame_max_time = history_key_frame_max_time;
  lco_param.history_key_frame_index = loop_history_index;
  lco_param.history_key_frame_fix_index =
      loop_history_index + history_key_frame_search_num_;

  SE3d history_key_frame_lidar_pose =
      trajectory_->GetLidarPose(lco_param.history_key_frame_timestamp);
  SE3d cur_key_frame_lidar_pose =
      trajectory_->GetLidarPose(lco_param.cur_key_frame_timestamp);
  //  SE3d correction_key_frame_lidar_pose =
  //      Matrix4fToSE3d(correction_transform) *
  //      cur_key_frame_lidar_pose;

  SE3d lidar_relative_pose = Matrix4fToSE3d(correction_transform);

  lco_param.loop_closure_edge.target_timestamp =
      lco_param.history_key_frame_timestamp;
  lco_param.loop_closure_edge.target_kf_index =
      lco_param.history_key_frame_index;
  lco_param.loop_closure_edge.source_timestamp =
      lco_param.cur_key_frame_timestamp;
  lco_param.loop_closure_edge.source_kf_index = lco_param.cur_key_frame_index;
  lco_param.loop_closure_edge.position = lidar_relative_pose.translation();
  lco_param.loop_closure_edge.orientation = lidar_relative_pose.so3();

  ComputeLoopClosureParam(lco_param);

  // 离散闭环优化
  DiscretedPoseGraph(lco_param);

  /// 闭环优化
  trajectory_manager_->OptimiseLoopClosure(lco_param, lc_weights_);

  SE3d delta_pose_before_optimization =
      history_key_frame_lidar_pose.inverse() * cur_key_frame_lidar_pose;
  SE3d history_pose =
      trajectory_->GetLidarPose(lco_param.history_key_frame_timestamp);
  SE3d cur_pose = trajectory_->GetLidarPose(lco_param.cur_key_frame_timestamp);
  SE3d delta_pose_after_optimization = history_pose.inverse() * cur_pose;
  Eigen::Vector3d his_bef_p, his_bef_r, cur_bef_p, cur_bef_r, delta_bef_p,
      delta_bef_r;
  Eigen::Vector3d his_aft_p, his_aft_r, cur_aft_p, cur_aft_r, delta_aft_p,
      delta_aft_r;
  SE3dToPositionEuler(history_key_frame_lidar_pose, his_bef_p, his_bef_r);
  SE3dToPositionEuler(cur_key_frame_lidar_pose, cur_bef_p, cur_bef_r);
  SE3dToPositionEuler(delta_pose_before_optimization, delta_bef_p, delta_bef_r);
  SE3dToPositionEuler(history_pose, his_aft_p, his_aft_r);
  SE3dToPositionEuler(cur_pose, cur_aft_p, cur_aft_r);
  SE3dToPositionEuler(delta_pose_after_optimization, delta_aft_p, delta_aft_r);
  std::cout << GREEN << "---------Loop Closure Result-----------" << std::endl;
  std::cout << lco_param.history_key_frame_timestamp << " <-----> "
            << lco_param.cur_key_frame_timestamp << std::endl;
  std::cout << "Before Loop Closure Optimization " << std::endl;
  std::cout << "History pose : " << his_bef_p.transpose() << "  ---  "
            << his_bef_r.transpose() << std::endl;
  std::cout << "Current pose : " << cur_bef_p.transpose() << "  ---  "
            << cur_bef_r.transpose() << std::endl;
  std::cout << "Delta pose : " << delta_bef_p.transpose() << "  ---  "
            << delta_bef_r.transpose() << std::endl;

  std::cout << "After Loop Closure Optimization " << std::endl;
  std::cout << "History pose : " << his_aft_p.transpose() << "  ---  "
            << his_aft_r.transpose() << std::endl;
  std::cout << "Current pose : " << cur_aft_p.transpose() << "  ---  "
            << cur_aft_r.transpose() << std::endl;
  std::cout << "Delta pose : " << delta_aft_p.transpose() << "  ---  "
            << delta_aft_r.transpose() << std::endl;

  std::cout << "-------------------------------------" << RESET << std::endl;

  history_loop_info_[loop_cur_index] = loop_history_index;
  history_loop_edgs_[loop_cur_index] = lco_param.loop_closure_edge;

  /// update key_pose
  for (size_t i = 0; i < cloud_key_pos_->size(); i++) {
    double kf_time = cloud_key_pos_->points[i].timestamp;
    if (kf_time < lco_param.history_key_frame_timestamp ||
        kf_time > lco_param.cur_key_frame_max_time) {
      continue;
    }

    Eigen::Vector3d p = trajectory_->GetLidarPose(kf_time).translation();
    cloud_key_pos_->points[i].x = p(0);
    cloud_key_pos_->points[i].y = p(1);
    cloud_key_pos_->points[i].z = p(2);
    cloud_key_pos_xy_->points[i].x = p(0);
    cloud_key_pos_xy_->points[i].y = p(1);
  }

  update_key_frame_ = true;

  //  std::cout << "Pause " << std::endl;
  //  std::getchar();
}

template <int _N>
void LidarOdometry<_N>::ExtractSurroundFeatures(const double cur_time) {
  if (cloud_key_pos_->points.empty()) return;

  /// Candidate keyframes
  PosCloud::Ptr key_pos_selected(new PosCloud);
  {
    /// Add the closest
    PosCloud::Ptr nearby_key_pos(new PosCloud);
    std::unordered_map<double, int> timestamp_hashmap;
    std::vector<int> point_search_ind;
    std::vector<float> point_search_sqr_distance;
    pcl::KdTreeFLANN<PosPoint>::Ptr kdtree_nearby_key_pos(
        new pcl::KdTreeFLANN<PosPoint>());
    kdtree_nearby_key_pos->setInputCloud(cloud_key_pos_);
    kdtree_nearby_key_pos->radiusSearch(
        cloud_key_pos_->back(), (double)keyframe_search_radius_,
        point_search_ind, point_search_sqr_distance);
    for (const int& idx : point_search_ind) {
      nearby_key_pos->push_back(cloud_key_pos_->points[idx]);
    }
    if (cloud_key_pos_->size() > 20) {
      surrounding_key_pos_voxel_filter_.SetInputCloud(nearby_key_pos);
      surrounding_key_pos_voxel_filter_.Filter(key_pos_selected);
    } else {
      key_pos_selected = nearby_key_pos;
    }

    for (size_t i = 0; i < key_pos_selected->size(); i++) {
      timestamp_hashmap.insert(
          std::pair<double, int>(key_pos_selected->points[i].timestamp, 1));
    }

    /// Add time nearest
    for (int i = cloud_key_pos_->size() - 1; i >= 0; i--) {
      if (cur_time - cloud_key_pos_->points[i].timestamp < 10.0) {
        auto iter = timestamp_hashmap.find(cloud_key_pos_->points[i].timestamp);
        if (iter == timestamp_hashmap.end())
          key_pos_selected->push_back(cloud_key_pos_->points[i]);
      } else
        break;
    }
  }

  feature_map_.Clear();

  double target_time = cloud_key_pos_->back().timestamp;
  feature_map_.timestamp = target_time;

  if (local_feature_container_.size() == 1) {
    if (use_corner_feature_) {
      *feature_map_.corner_features +=
          *(local_feature_container_[target_time].corner_features);
    }
    *feature_map_.surface_features +=
        *(local_feature_container_[target_time].surface_features);
    *feature_map_.full_features +=
        *(local_feature_container_[target_time].full_features);
    feature_map_.time_max = local_feature_container_[target_time].time_max;
  } else {
    SE3d pose_G_to_target = trajectory_->GetLidarPose(target_time).inverse();

    for (auto const& pos : key_pos_selected->points) {
      double kf_time = pos.timestamp;
      if (local_feature_container_.find(kf_time) !=
          local_feature_container_.end()) {
        // transform local scan to the frame of last keyframe
        SE3d pose_cur_to_G = trajectory_->GetLidarPose(kf_time);
        Eigen::Matrix4d cur_to_target =
            (pose_G_to_target * pose_cur_to_G).matrix();

        if (use_corner_feature_) {
          PosCloud::Ptr corner_in_target(new PosCloud());
          pcl::transformPointCloud(
              *(local_feature_container_[kf_time].corner_features),
              *corner_in_target, cur_to_target);
          *feature_map_.corner_features += *corner_in_target;
        }

        PosCloud::Ptr surface_in_target(new PosCloud());
        PosCloud::Ptr full_in_target(new PosCloud());

        pcl::transformPointCloud(
            *(local_feature_container_[kf_time].surface_features),
            *surface_in_target, cur_to_target);
        pcl::transformPointCloud(
            *(local_feature_container_[kf_time].full_features), *full_in_target,
            cur_to_target);

        *feature_map_.surface_features += *surface_in_target;
        *feature_map_.full_features += *full_in_target;
        feature_map_.time_max = std::max(
            feature_map_.time_max, local_feature_container_[kf_time].time_max);
      }
    }
  }

  DownsampleCurrentScan(feature_map_, feature_map_ds_);

  feature_map_ds_.timestamp = target_time;
  SetTargetFeature(feature_map_ds_);
}

template <int _N>
void LidarOdometry<_N>::DownsampleCurrentScan(LiDARFeature& kf_in,
                                              LiDARFeature& kf_out) {
  kf_out.Clear();
  kf_out.timestamp = kf_in.timestamp;
  kf_out.time_max = kf_in.time_max;

  if (use_corner_feature_) {
    corner_features_voxel_filter_.SetInputCloud(kf_in.corner_features);
    corner_features_voxel_filter_.Filter(kf_out.corner_features);
    //    kf_out.corner_features = corner_feature_filter;
  }

  surface_features_voxel_filter_.SetInputCloud(kf_in.surface_features);
  surface_features_voxel_filter_.Filter(kf_out.surface_features);
  //  kf_out.surface_features = surface_feature_filter;
}

template <int _N>
const bool LidarOdometry<_N>::isKeyFrame(double time) const {
  // The first scan
  if (cloud_key_pos_->points.empty()) return true;

  if (time > trajectory_->GetActiveTime()) return false;

  if ((time - cloud_key_pos_->back().timestamp) > 10) return true;

  SE3d pose_cur = trajectory_->GetLidarPose(time);
  SE3d pose_last = trajectory_->GetLidarPose(cloud_key_pos_->back().timestamp);

  SE3d pose_cur_to_last = pose_last.inverse() * pose_cur;
  Eigen::AngleAxisd v(pose_cur_to_last.so3().unit_quaternion());

  double dist_meter = pose_cur_to_last.translation().norm();
  double angle_degree = v.angle() * (180 / M_PI);

  if (angle_degree > keyframe_adding_angle_threshold_ ||
      dist_meter > keyframe_adding_dist_meter_) {
    //    std::cout << WHITE << "[ isKeyFrame ] " << angle_degree << " : "
    //              << dist_meter << RESET << std::endl;
    return true;
  }

  return false;
}

template <int _N>
void LidarOdometry<_N>::UpdateKeyFrames() {
  /// All current scan to cache
  LiDARFeature feature;
  feature = feature_cur_;
  cache_feature_container_[feature.timestamp] = feature;

  for (auto iter = cache_feature_container_.begin();
       iter != cache_feature_container_.end();) {
    if (!isKeyFrame(iter->second.time_max)) {
      if (iter->second.time_max > trajectory_->GetActiveTime())
        break;
      else {
        // remove non-keyframe
        cache_feature_container_.erase(iter++);
        continue;
      }
    }

    LiDARFeature feature;
    feature.timestamp = iter->second.timestamp;
    feature.time_max = iter->second.time_max;

    PosPoint key_pose;
    key_pose.timestamp = feature.timestamp;
    Eigen::Vector3d p;
    p = trajectory_->GetLidarPose(feature.timestamp).translation();
    key_pose.x = p[0];
    key_pose.y = p[1];
    key_pose.z = p[2];

    if (local_feature_container_.empty()) {
      // key_pose.timestamp = trajectory_->minTime();

      *(feature.corner_features) = *(iter->second.corner_features);
      *(feature.surface_features) = *(iter->second.surface_features);
      *(feature.full_features) = *(iter->second.full_features);

    } else {
      /// transfrom each point to the frame of first point
      trajectory_->UndistortScan(*(iter->second.corner_features),
                                 feature.timestamp, *(feature.corner_features));
      trajectory_->UndistortScan(*(iter->second.surface_features),
                                 feature.timestamp,
                                 *(feature.surface_features));
      trajectory_->UndistortScan(*(iter->second.full_features),
                                 feature.timestamp, *(feature.full_features));
    }

    trajectory_->SetForcedFixedTime(feature.time_max);

    cloud_key_pos_->push_back(key_pose);
    local_feature_container_[key_pose.timestamp] = feature;

    key_pose.z = 0;
    cloud_key_pos_xy_->push_back(key_pose);

    cache_feature_container_.erase(iter++);
    update_key_frame_ = true;
  }
}

template <int _N>
void LidarOdometry<_N>::PointCloudConvert(const RTPointCloud::Ptr input_cloud,
                                          double scan_timestamp,
                                          PosCloud::Ptr output_cloud,
                                          double* max_time) {
  output_cloud->header = input_cloud->header;
  output_cloud->resize(input_cloud->size());
  if (max_time) *max_time = scan_timestamp;

  size_t cnt = 0;
  for (auto const& v : input_cloud->points) {
    if (v.time < 0) continue;
    PosPoint p;
    p.x = v.x;
    p.y = v.y;
    p.z = v.z;

    p.timestamp = scan_timestamp + v.time;
    output_cloud->points[cnt++] = p;

    if (max_time && (*max_time < p.timestamp)) *max_time = p.timestamp;
  }
  // Resize to the correct size
  if (cnt != input_cloud->size()) output_cloud->resize(cnt);
  //  std::cout << "output cloud size: " << output_cloud->size() << std::endl;
}

template <int _N>
bool LidarOdometry<_N>::SetTargetFeature(const LiDARFeature& feature_map) {
  if ((use_corner_feature_ && feature_map.corner_features->empty()) ||
      feature_map.surface_features->empty()) {
    return false;
  }
  if (use_corner_feature_)
    kdtree_corner_map_->setInputCloud(feature_map.corner_features);
  kdtree_surface_map_->setInputCloud(feature_map.surface_features);
  return true;
}

template <int _N>
bool LidarOdometry<_N>::FindCorrespondence(
    const LiDARFeature& feature_cur, const LiDARFeature& feature_cur_in_M) {
  point_correspondence_.clear();

  if ((use_corner_feature_ &&
       (feature_cur.corner_features->size() < edge_min_valid_num_)) ||
      feature_cur.surface_features->size() < surf_min_valid_num_) {
    //    std::cout << RED
    //              << "[FindCorrespondence] No enough feature points ! Corner :
    //              "
    //              << feature_cur.corner_features->size()
    //              << "  Surface : " << feature_cur.surface_features->size() <<
    //              RESET
    //              << std::endl;
  }

  if (use_corner_feature_) {
    for (size_t i = 0; i < feature_cur.corner_features->size(); i++) {
      PosPoint point_global = feature_cur_in_M.corner_features->points[i];
      std::vector<int> point_search_id;
      std::vector<float> point_search_dis;
      kdtree_corner_map_->nearestKSearch(point_global, 5, point_search_id,
                                         point_search_dis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

      if (point_search_dis[4] < 1.0) {
        Eigen::Vector3d cp(0, 0, 0);
        for (int j = 0; j < 5; j++) {
          cp[0] +=
              feature_map_ds_.corner_features->points[point_search_id[j]].x;
          cp[1] +=
              feature_map_ds_.corner_features->points[point_search_id[j]].y;
          cp[2] +=
              feature_map_ds_.corner_features->points[point_search_id[j]].z;
        }
        cp /= 5.0;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax =
              feature_map_ds_.corner_features->points[point_search_id[j]].x -
              cp[0];
          float ay =
              feature_map_ds_.corner_features->points[point_search_id[j]].y -
              cp[1];
          float az =
              feature_map_ds_.corner_features->points[point_search_id[j]].z -
              cp[2];

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        cv::eigen(matA1, matD1, matV1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
          Eigen::Vector3d normal(matV1.at<float>(0, 0), matV1.at<float>(0, 1),
                                 matV1.at<float>(0, 2));
          double plane_d = -normal.dot(cp);

          PointCorrespondence point_cor;
          point_cor.geo_type = Line;
          point_cor.t_point = feature_cur.corner_features->points[i].timestamp;
          point_cor.t_map = feature_map_.timestamp;
          point_cor.geo_point = cp;
          point_cor.geo_normal = normal;
          point_cor.point =
              Eigen::Vector3d(feature_cur.corner_features->points[i].x,
                              feature_cur.corner_features->points[i].y,
                              feature_cur.corner_features->points[i].z);
          point_correspondence_.push_back(point_cor);
        }
      }
    }
  }

  for (size_t i = 0; i < feature_cur.surface_features->size(); i++) {
    if (feature_cur.surface_features->points[i].timestamp < 0) continue;

    PosPoint point_global = feature_cur_in_M.surface_features->points[i];
    std::vector<int> point_search_id;
    std::vector<float> point_search_dis;
    kdtree_surface_map_->nearestKSearch(point_global, 5, point_search_id,
                                        point_search_dis);

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    if (point_search_dis[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) =
            feature_map_ds_.surface_features->points[point_search_id[j]].x;
        matA0(j, 1) =
            feature_map_ds_.surface_features->points[point_search_id[j]].y;
        matA0(j, 2) =
            feature_map_ds_.surface_features->points[point_search_id[j]].z;
      }
      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (fabs(pa * feature_map_ds_.surface_features
                          ->points[point_search_id[j]]
                          .x +
                 pb * feature_map_ds_.surface_features
                          ->points[point_search_id[j]]
                          .y +
                 pc * feature_map_ds_.surface_features
                          ->points[point_search_id[j]]
                          .z +
                 pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        PointCorrespondence point_cor;
        point_cor.geo_type = Plane;
        point_cor.t_point = feature_cur.surface_features->points[i].timestamp;
        point_cor.t_map = feature_map_ds_.timestamp;
        point_cor.point =
            Eigen::Vector3d(feature_cur.surface_features->points[i].x,
                            feature_cur.surface_features->points[i].y,
                            feature_cur.surface_features->points[i].z);
        point_cor.geo_plane = Eigen::Vector4d(pa, pb, pc, pd);
        point_correspondence_.push_back(point_cor);
      }
    }
  }

  return true;
}

template <int _N>
bool LidarOdometry<_N>::ScanMatch(int iter_num) {
  LiDARFeature feature_cur_in_map;

  int iter_count = 0;
  for (; iter_count < iter_num; iter_count++) {
    if (use_corner_feature_) {
      trajectory_->UndistortScan(*feature_cur_ds_.corner_features,
                                 feature_map_ds_.timestamp,
                                 *feature_cur_in_map.corner_features);
    }

    trajectory_->UndistortScan(*feature_cur_ds_.surface_features,
                               feature_map_ds_.timestamp,
                               *feature_cur_in_map.surface_features);

    FindCorrespondence(feature_cur_ds_, feature_cur_in_map);
    DownSampleCorrespondence();

    trajectory_manager_->BuildProblemAndSolve(point_correspondence_, 50);
  }

  if (use_corner_feature_) {
    trajectory_->UndistortScan(*feature_cur_ds_.corner_features,
                               feature_map_ds_.timestamp,
                               *feature_cur_in_map.corner_features);
  }

  trajectory_->UndistortScan(*feature_cur_ds_.surface_features,
                             feature_map_ds_.timestamp,
                             *feature_cur_in_map.surface_features);

  FindCorrespondence(feature_cur_ds_, feature_cur_in_map);

  DownSampleCorrespondence();

  trajectory_manager_->BuildProblemAndSolve(point_correspondence_, 50);

  trajectory_manager_->UpdateTrajectoryProperty();

  //  trajectory_manager_->EstimateIMUBias();
  return true;
}

template <int _N>
void LidarOdometry<_N>::SaveKeyFrameCloud(std::string path) {
  PosCloud full_cloud, surface_cloud, corner_cloud;
  /// check each keyframe
  for (PosPoint p : cloud_key_pos_->points) {
    double kf_time = p.timestamp;
    if (local_feature_container_.find(kf_time) !=
        local_feature_container_.end()) {
      SE3d pose_cur_to_G = trajectory_->GetLidarPose(kf_time);
      Eigen::Matrix4d pose_matrix = pose_cur_to_G.matrix();
      if (use_corner_feature_) {
        PosCloud::Ptr corner_in_G(new PosCloud());
        pcl::transformPointCloud(
            *(local_feature_container_[kf_time].corner_features), *corner_in_G,
            pose_matrix);
        corner_cloud += *corner_in_G;
      }

      PosCloud::Ptr surface_in_G(new PosCloud());
      pcl::transformPointCloud(
          *(local_feature_container_[kf_time].surface_features), *surface_in_G,
          pose_matrix);
      surface_cloud += *surface_in_G;

      PosCloud::Ptr full_in_G(new PosCloud());
      pcl::transformPointCloud(
          *(local_feature_container_[kf_time].full_features), *full_in_G,
          pose_matrix);
      full_cloud += *full_in_G;
    }
  }

  VPointCloud full_map, surface_map, corner_map, feature_map;
  PosCloudConvert(full_cloud, full_map);
  PosCloudConvert(surface_cloud, surface_map);
  if (use_corner_feature_) {
    PosCloudConvert(corner_cloud, corner_map);
    feature_map += corner_map;
  }
  feature_map += surface_map;

  // save pcd
  pcl::io::savePCDFileBinaryCompressed(path + "/full_map.pcd", full_map);
  pcl::io::savePCDFileBinaryCompressed(path + "/feature_map.pcd", feature_map);

  std::cout << "Save cloud at " << path + "/full_map.pcd"
            << "; size: " << full_map.size() << std::endl;
  std::cout << "Save cloud at " << path + "/feature_map.pcd"
            << "; size: " << feature_map.size() << std::endl;
}

template <int _N>
void LidarOdometry<_N>::PosCloudConvert(const PosCloud& input_cloud,
                                        VPointCloud& output_cloud) {
  for (size_t i = 0; i < input_cloud.size(); i++) {
    VPoint p;
    p.x = input_cloud.points[i].x;
    p.y = input_cloud.points[i].y;
    p.z = input_cloud.points[i].z;
    output_cloud.push_back(p);
  }
}

template <int _N>
void LidarOdometry<_N>::DownSampleCorrespondence() {
  if (downsample_num_ <= 1) return;
  int cnt = 0;
  auto iter = point_correspondence_.begin();
  while (iter != point_correspondence_.end()) {
    if ((cnt++) % downsample_num_ == 0) {
      iter++;
    } else {
      iter = point_correspondence_.erase(iter);
    }
  }
}

template <int _N>
bool LidarOdometry<_N>::DetectLoopClosure(int& latest_index,
                                          int& closest_index) {
  int loop_key_cur = cloud_key_pos_xy_copied_->size() - 1;
  int loop_key_history = -1;
  double cur_timestamp =
      cloud_key_pos_xy_copied_->points[loop_key_cur].timestamp;

  auto iter = history_loop_info_.find(loop_key_cur);
  if (iter != history_loop_info_.end()) {
    return false;
  }

  std::vector<int> loop_search_index;
  std::vector<float> loop_search_distance;
  kdtree_history_key_poses_->setInputCloud(cloud_key_pos_xy_copied_);
  kdtree_history_key_poses_->radiusSearch(
      cloud_key_pos_xy_copied_->back(), history_key_frame_search_radius_,
      loop_search_index, loop_search_distance);

  for (int i = 0; i < loop_search_index.size(); i++) {
    int index = loop_search_index[i];
    if (fabs(cloud_key_pos_xy_copied_->points[index].timestamp -
             cur_timestamp) > history_key_frame_time_diff_ &&
        (loop_key_cur - index) > history_key_frame_index_diff_) {
      loop_key_history = index;
      break;
    }
  }

  if (loop_key_history == -1 || loop_key_cur == loop_key_history) {
    return false;
  }

  latest_index = loop_key_cur;
  closest_index = loop_key_history;
  return true;
}

template <int _N>
void LidarOdometry<_N>::FindNearbyKeyFrames(GPointCloud::Ptr cloud,
                                            const int key, const int search_num,
                                            double& min_time,
                                            double& max_time) {
  cloud->clear();
  int cloud_size = cloud_key_pos_xy_copied_->size();
  for (int i = -search_num; i <= search_num; i++) {
    int key_index = key + i;
    if (key_index < 0 || key_index >= cloud_size) continue;

    double kf_time = cloud_key_pos_->points[key_index].timestamp;
    if (local_feature_container_.find(kf_time) !=
        local_feature_container_.end()) {
      SE3d pose_cur_to_G = trajectory_->GetLidarPose(kf_time);
      Eigen::Matrix4d global_pose = pose_cur_to_G.matrix();
      GPointCloud local_cloud;

      if (i == -search_num) {
        min_time = local_feature_container_[kf_time].timestamp;
      }
      if (i == search_num) {
        max_time = local_feature_container_[kf_time].time_max;
      }

      if (local_feature_container_[kf_time].full_features->size() != 0 &&
          loop_closure_use_full_cloud_) {
        for (size_t i = 0;
             i < local_feature_container_[kf_time].full_features->size(); i++) {
          GPoint p;
          p.x = local_feature_container_[kf_time].full_features->points[i].x;
          p.y = local_feature_container_[kf_time].full_features->points[i].y;
          p.z = local_feature_container_[kf_time].full_features->points[i].z;
          local_cloud.push_back(p);
        }
      } else {
        for (size_t i = 0;
             i < local_feature_container_[kf_time].surface_features->size();
             i++) {
          GPoint p;
          p.x = local_feature_container_[kf_time].surface_features->points[i].x;
          p.y = local_feature_container_[kf_time].surface_features->points[i].y;
          p.z = local_feature_container_[kf_time].surface_features->points[i].z;
          local_cloud.push_back(p);
        }
        for (size_t i = 0;
             i < local_feature_container_[kf_time].corner_features->size();
             i++) {
          GPoint p;
          p.x = local_feature_container_[kf_time].corner_features->points[i].x;
          p.y = local_feature_container_[kf_time].corner_features->points[i].y;
          p.z = local_feature_container_[kf_time].corner_features->points[i].z;
          local_cloud.push_back(p);
        }
      }

      GPointCloud global_cloud;
      pcl::transformPointCloud(local_cloud, global_cloud, global_pose);
      *cloud += global_cloud;
    }
  }

  GPointCloud::Ptr cloud_temp(new GPointCloud);
  loop_closure_match_voxel_filter_.setInputCloud(cloud);
  loop_closure_match_voxel_filter_.filter(*cloud_temp);
  *cloud = *cloud_temp;
}

template <int _N>
void LidarOdometry<_N>::FindNearbyKeyFrames(GPointCloud::Ptr cloud,
                                            const int key, const int search_num,
                                            double& min_time, double& max_time,
                                            Eigen::Matrix4d& pose_in_G) {
  cloud->clear();
  int cloud_size = cloud_key_pos_xy_copied_->size();

  double target_kf_time = cloud_key_pos_->points[key].timestamp;
  SE3d target_pose_to_G = trajectory_->GetLidarPose(target_kf_time);
  pose_in_G = target_pose_to_G.matrix();

  for (int i = -search_num; i <= search_num; i++) {
    int key_index = key + i;
    if (key_index < 0 || key_index >= cloud_size) continue;

    double kf_time = cloud_key_pos_->points[key_index].timestamp;
    if (local_feature_container_.find(kf_time) !=
        local_feature_container_.end()) {
      SE3d pose_cur_to_target =
          target_pose_to_G.inverse() * trajectory_->GetLidarPose(kf_time);
      Eigen::Matrix4d local_pose = pose_cur_to_target.matrix();
      GPointCloud local_cloud;

      if (i == -search_num) {
        min_time = local_feature_container_[kf_time].timestamp;
      }
      if (i == search_num) {
        max_time = local_feature_container_[kf_time].time_max;
      }

      if (local_feature_container_[kf_time].full_features->size() != 0 &&
          loop_closure_use_full_cloud_) {
        for (size_t i = 0;
             i < local_feature_container_[kf_time].full_features->size(); i++) {
          GPoint p;
          p.x = local_feature_container_[kf_time].full_features->points[i].x;
          p.y = local_feature_container_[kf_time].full_features->points[i].y;
          p.z = local_feature_container_[kf_time].full_features->points[i].z;
          local_cloud.push_back(p);
        }
      } else {
        for (size_t i = 0;
             i < local_feature_container_[kf_time].surface_features->size();
             i++) {
          GPoint p;
          p.x = local_feature_container_[kf_time].surface_features->points[i].x;
          p.y = local_feature_container_[kf_time].surface_features->points[i].y;
          p.z = local_feature_container_[kf_time].surface_features->points[i].z;
          local_cloud.push_back(p);
        }
        for (size_t i = 0;
             i < local_feature_container_[kf_time].corner_features->size();
             i++) {
          GPoint p;
          p.x = local_feature_container_[kf_time].corner_features->points[i].x;
          p.y = local_feature_container_[kf_time].corner_features->points[i].y;
          p.z = local_feature_container_[kf_time].corner_features->points[i].z;
          local_cloud.push_back(p);
        }
      }

      GPointCloud global_cloud;
      pcl::transformPointCloud(local_cloud, global_cloud, local_pose);
      *cloud += global_cloud;
    }
  }

  GPointCloud::Ptr cloud_temp(new GPointCloud);
  loop_closure_match_voxel_filter_.setInputCloud(cloud);
  loop_closure_match_voxel_filter_.filter(*cloud_temp);
  *cloud = *cloud_temp;
}

template <int _N>
double LidarOdometry<_N>::ComputePoseHeightDiff(int target_index,
                                                int source_index) {
  double target_kf_time = cloud_key_pos_->points[target_index].timestamp;
  double source_kf_time = cloud_key_pos_->points[source_index].timestamp;

  Eigen::Vector3d target_global_trans =
      trajectory_->GetLidarPose(target_kf_time).translation();
  Eigen::Vector3d source_global_trans =
      trajectory_->GetLidarPose(source_kf_time).translation();

  double delta_height = source_global_trans(2) - target_global_trans(2);
  return delta_height;
}

template <int _N>
void LidarOdometry<_N>::CacheLoopICPResult(GPointCloud::Ptr target_cloud,
                                           GPointCloud::Ptr source_cloud,
                                           Eigen::Matrix4f& guess,
                                           Eigen::Matrix4f& icp_reuslt) {
  std::string loop_cache_path = cache_path_ + "/loop_closure_icp";
  if (!boost::filesystem::exists(loop_cache_path))
    boost::filesystem::create_directory(loop_cache_path);

  int loop_index = history_loop_info_.size();
  std::stringstream ss;
  ss << loop_index;
  pcl::io::savePCDFileBinaryCompressed(
      loop_cache_path + "/target_cloud_" + ss.str() + ".pcd", *target_cloud);
  GPointCloud raw_source_cloud, icp_source_cloud;
  pcl::transformPointCloud(*source_cloud, raw_source_cloud, guess);
  pcl::transformPointCloud(*source_cloud, icp_source_cloud, icp_reuslt);
  pcl::io::savePCDFileBinaryCompressed(
      loop_cache_path + "/raw_source_cloud_" + ss.str() + ".pcd",
      raw_source_cloud);
  pcl::io::savePCDFileBinaryCompressed(
      loop_cache_path + "/icp_source_cloud_" + ss.str() + ".pcd",
      icp_source_cloud);
}

template <int _N>
void LidarOdometry<_N>::ComputeLoopClosureParam(
    LoopClosureOptimizationParam& param) {
  double loop_start_time = param.history_key_frame_timestamp;
  double loop_end_time = param.cur_key_frame_max_time;
  int loop_kf_start_index = param.history_key_frame_index;
  int loop_kf_end_index = param.cur_key_frame_index;

  param.pose_graph_start_index = loop_kf_start_index;

  for (auto iter = history_loop_edgs_.begin(); iter != history_loop_edgs_.end();
       iter++) {
    if ((iter->second.target_kf_index >= loop_kf_start_index &&
         iter->second.target_kf_index <= loop_kf_end_index) ||
        (iter->second.source_kf_index > loop_kf_start_index &&
         iter->second.source_kf_index < loop_kf_end_index)) {
      param.history_loop_closure_edges.push_back(iter->second);
      param.pose_graph_start_index =
          std::min(param.pose_graph_start_index, iter->second.target_kf_index);
    }
  }

  for (double t = loop_start_time; t < feature_cur_.time_max; t += 0.05) {
    SE3d pose = trajectory_->pose(t);
    Eigen::Vector3d vel_in_w = trajectory_->transVelWorld(t);
    Eigen::Vector3d gyro = trajectory_->rotVelBody(t);
    PoseData imu_pose;
    imu_pose.timestamp = t;
    imu_pose.position = pose.translation();
    imu_pose.orientation = pose.so3();

    VelocityData vd;
    vd.timestamp = t;
    vd.velocity = pose.so3().inverse() * vel_in_w;
    vd.gyro = gyro;

    param.velocity_constraint.push_back(vd);
  }
}

template <int _N>
void LidarOdometry<_N>::DiscretedPoseGraph(
    LoopClosureOptimizationParam& param) {
  using namespace ceres::examples;
  MapOfPoses graph_poses;
  VectorOfConstraints graph_constraints;

  std::vector<SE3d> pose_vector;
  std::vector<int> fix_index;
  int index = param.pose_graph_start_index;
  int cnt = 0;
  for (; index <= param.cur_key_frame_index; index++) {
    double kf_time = cloud_key_pos_->points[index].timestamp;
    SE3d lidar_pose = trajectory_->GetLidarPose(kf_time);
    pose_vector.push_back(lidar_pose);
    Pose3d pose_temp;
    pose_temp.p = lidar_pose.translation();
    pose_temp.q = lidar_pose.unit_quaternion();
    graph_poses.insert(std::make_pair(cnt, pose_temp));
    if (index >= param.history_key_frame_index &&
        index <= param.history_key_frame_fix_index) {
      fix_index.push_back(cnt);
    }
    cnt++;
  }

  double trajectory_max_time = feature_cur_.time_max;
  SE3d last_lidar_pose = trajectory_->GetLidarPose(trajectory_max_time);
  pose_vector.push_back(last_lidar_pose);
  Pose3d last_pose;
  last_pose.p = last_lidar_pose.translation();
  last_pose.q = last_lidar_pose.unit_quaternion();
  graph_poses.insert(std::make_pair(cnt, last_pose));

  for (size_t i = 0; i < pose_vector.size() - 1; i++) {
    Constraint3d temp;
    temp.id_begin = i;
    temp.id_end = i + 1;
    temp.information << 10000, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 10000,
        0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 0, 0,
        20000;
    SE3d delta_pose = pose_vector[i].inverse() * pose_vector[i + 1];
    temp.t_be.p = delta_pose.translation();
    temp.t_be.q = delta_pose.unit_quaternion();
    graph_constraints.push_back(temp);
  }

  Constraint3d temp;
  temp.id_begin = param.history_key_frame_index - param.pose_graph_start_index;
  temp.id_end = param.cur_key_frame_index - param.pose_graph_start_index;
  temp.information << 10000, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 10000,
      0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 0, 0, 20000, 0, 0, 0, 0, 0, 0, 20000;
  temp.t_be.p = param.loop_closure_edge.position;
  temp.t_be.q = param.loop_closure_edge.orientation.unit_quaternion();
  graph_constraints.push_back(temp);

  for (size_t i = 0; i < param.history_loop_closure_edges.size(); i++) {
    Constraint3d temp;
    temp.id_begin = param.history_loop_closure_edges[i].target_kf_index -
                    param.pose_graph_start_index;
    temp.id_end = param.history_loop_closure_edges[i].source_kf_index -
                  param.pose_graph_start_index;
    temp.information << 10000, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 10000,
        0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0, 0,
        10000;
    temp.t_be.p = param.history_loop_closure_edges[i].position;
    temp.t_be.q =
        param.history_loop_closure_edges[i].orientation.unit_quaternion();
    graph_constraints.push_back(temp);
  }

  /// Optimization
  ceres::Problem problem;

  ceres::LossFunction* loss_function = NULL;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (VectorOfConstraints::const_iterator constraints_iter =
           graph_constraints.begin();
       constraints_iter != graph_constraints.end(); ++constraints_iter) {
    const Constraint3d& constraint = *constraints_iter;

    MapOfPoses::iterator pose_begin_iter =
        graph_poses.find(constraint.id_begin);
    MapOfPoses::iterator pose_end_iter = graph_poses.find(constraint.id_end);

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem.AddResidualBlock(cost_function, loss_function,
                             pose_begin_iter->second.p.data(),
                             pose_begin_iter->second.q.coeffs().data(),
                             pose_end_iter->second.p.data(),
                             pose_end_iter->second.q.coeffs().data());

    problem.SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                quaternion_local_parameterization);
    problem.SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                quaternion_local_parameterization);
  }

  for (size_t i = 0; i < fix_index.size(); i++) {
    int index = fix_index[i];
    MapOfPoses::iterator iter = graph_poses.find(index);
    if (iter != graph_poses.end()) {
      problem.SetParameterBlockConstant(iter->second.p.data());
      problem.SetParameterBlockConstant(iter->second.q.coeffs().data());
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = std::thread::hardware_concurrency();

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  //  PosCloud full_cloud;
  std::vector<SE3d> pose_data_after_optimization;
  for (int id = param.history_key_frame_index; id <= param.cur_key_frame_index;
       id++) {
    int pose_index = id - param.pose_graph_start_index;
    auto pose_iter = graph_poses.find(pose_index);
    if (pose_iter == graph_poses.end()) continue;

    PoseData pd;
    pd.timestamp = cloud_key_pos_->points[id].timestamp;
    pd.position = pose_iter->second.p;
    pd.orientation = Sophus::SO3d(pose_iter->second.q);
    param.pose_graph_key_pose.push_back(pd);

    pose_data_after_optimization.push_back(SE3d(pd.orientation, pd.position));
  }

  auto pose_iter = graph_poses.find(cnt);
  PoseData pd;
  pd.timestamp = trajectory_max_time;
  pd.position = pose_iter->second.p;
  pd.orientation = Sophus::SO3d(pose_iter->second.q);
  param.pose_graph_key_pose.push_back(pd);

  pose_data_after_optimization.push_back(SE3d(pd.orientation, pd.position));

  TrajectoryViewer::PublishDiscretedPoseGraphMarker(
      pose_vector, pose_data_after_optimization);
}

template <int _N>
bool LidarOdometry<_N>::PointToPointICP(GPointCloud::Ptr target_cloud,
                                        GPointCloud::Ptr source_cloud,
                                        double correspondence_distance,
                                        Eigen::Matrix4f guess,
                                        Eigen::Matrix4f& transform,
                                        float& fitness_score) {
  if (correspondence_distance < 1) correspondence_distance = 1;
  pcl::IterativeClosestPoint<GPoint, GPoint> icp;
  icp.setMaxCorrespondenceDistance(correspondence_distance);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  icp.setInputSource(source_cloud);
  icp.setInputTarget(target_cloud);

  GPointCloud result_cloud;
  icp.align(result_cloud, guess);

  fitness_score = icp.getFitnessScore();
  transform = icp.getFinalTransformation();
  if (icp.hasConverged() == false ||
      fitness_score > history_key_frame_fitness_score_) {
    std::cout << "ICP failed ; icp score : " << fitness_score
              << " Coverged : " << icp.hasConverged() << std::endl;
    return false;
  }

  std::cout << GREEN << "Icp score : " << fitness_score
            << " Coverged : " << icp.hasConverged() << RESET << std::endl;
  std::cout << GREEN << "ICP transform matrix : " << std::endl;
  std::cout << transform << RESET << std::endl;

  std::cout << "---------------Loop Closure ICP Match------------------ "
            << std::endl;
  std::cout << "ICP Index : " << history_loop_info_.size() << std::endl;
  std::cout << "ICP Score : " << fitness_score
            << " Coverged : " << icp.hasConverged() << std::endl;
  std::cout << "ICP transform matrix : " << std::endl;
  std::cout << transform << std::endl;

  return true;
}

template <int _N>
bool LidarOdometry<_N>::PointToPlaneICP(GPointCloud::Ptr target_cloud,
                                        GPointCloud::Ptr source_cloud,
                                        Eigen::Matrix4f& transform,
                                        float& fitness_score) {
  pcl::PointCloud<pcl::PointNormal>::Ptr src(
      new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*source_cloud, *src);
  pcl::PointCloud<pcl::PointNormal>::Ptr tgt(
      new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*target_cloud, *tgt);

  pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
  norm_est.setSearchMethod(pcl::search::KdTree<pcl::PointNormal>::Ptr(
      new pcl::search::KdTree<pcl::PointNormal>));
  norm_est.setKSearch(50);
  norm_est.setInputCloud(tgt);
  norm_est.compute(*tgt);

  pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
  typedef pcl::registration::TransformationEstimationPointToPlane<
      pcl::PointNormal, pcl::PointNormal>
      PointToPlane;
  boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
  icp.setTransformationEstimation(point_to_plane);

  icp.setInputSource(src);
  icp.setInputTarget(tgt);

  icp.setRANSACIterations(0);
  icp.setMaximumIterations(60);
  icp.setTransformationEpsilon(1e-6);
  icp.setMaxCorrespondenceDistance(history_key_frame_search_radius_);
  pcl::PointCloud<pcl::PointNormal> output;
  icp.align(output);
  fitness_score = icp.getFitnessScore();
  if (icp.hasConverged() == false ||
      fitness_score > history_key_frame_fitness_score_) {
    std::cout << "ICP failed ; icp score : " << fitness_score
              << " Coverged : " << icp.hasConverged() << std::endl;
    return false;
  }

  transform = icp.getFinalTransformation();

  std::cout << GREEN << "Icp score : " << fitness_score
            << " Coverged : " << icp.hasConverged() << RESET << std::endl;
  std::cout << GREEN << "ICP transform matrix : " << std::endl;
  std::cout << transform << RESET << std::endl;

  std::cout << "---------------Loop Closure ICP Match------------------ "
            << std::endl;
  std::cout << "ICP Index : " << history_loop_info_.size() << std::endl;
  std::cout << "ICP Score : " << fitness_score
            << " Coverged : " << icp.hasConverged() << std::endl;
  std::cout << "ICP transform matrix : " << std::endl;
  std::cout << transform << std::endl;

  return true;
}

}  // namespace clins

#endif
