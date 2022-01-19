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

#ifndef TRAJECTORY_VIEWER_HPP
#define TRAJECTORY_VIEWER_HPP

#include <rosbag/bag.h>

#include <sensor_data/imu_data.h>
#include <trajectory/se3_trajectory.hpp>

#include <clins/imu_array.h>
#include <clins/pose_array.h>
#include <eigen_conversions/eigen_msg.h>
#include <feature/lidar_feature.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Eigen>

namespace clins {

namespace publisher {
static ros::Publisher pub_trajectory_raw_;
static ros::Publisher pub_trajectory_est_;
static ros::Publisher pub_imu_raw_array_;
static ros::Publisher pub_imu_est_array_;

// lidar odometry
static ros::Publisher pub_target_dense_cloud_;
static ros::Publisher pub_source_dense_cloud_;

// pose graph
static ros::Publisher pub_icp_target_cloud_;
static ros::Publisher pub_icp_source_cloud_;
static ros::Publisher pub_icp_raw_source_cloud_;
static ros::Publisher pub_pose_graph_marker_;

static ros::Publisher pub_spline_trajectory_;

static void SetPublisher(ros::NodeHandle &nh) {
  /// Vicon data
  pub_trajectory_raw_ = nh.advertise<clins::pose_array>("/path_raw", 10);
  pub_trajectory_est_ = nh.advertise<clins::pose_array>("/path_est", 10);
  /// IMU fitting results
  pub_imu_raw_array_ = nh.advertise<clins::imu_array>("/imu_raw_array", 10);
  pub_imu_est_array_ = nh.advertise<clins::imu_array>("/imu_est_array", 10);

  /// spline trajectory
  pub_spline_trajectory_ =
      nh.advertise<nav_msgs::Path>("/spline_trajectory", 10);

  /// dense feature cloud
  pub_target_dense_cloud_ =
      nh.advertise<sensor_msgs::PointCloud2>("/target_dense_cloud", 10);
  pub_source_dense_cloud_ =
      nh.advertise<sensor_msgs::PointCloud2>("/source_dense_cloud", 10);

  /// publish icp cloud
  pub_icp_target_cloud_ =
      nh.advertise<sensor_msgs::PointCloud2>("/icp_target_cloud", 10);
  pub_icp_source_cloud_ =
      nh.advertise<sensor_msgs::PointCloud2>("/icp_source_cloud", 10);
  pub_icp_raw_source_cloud_ =
      nh.advertise<sensor_msgs::PointCloud2>("/icp_raw_source_cloud", 10);

  pub_pose_graph_marker_ =
      nh.advertise<visualization_msgs::MarkerArray>("/pose_graph_markers", 10);
}
}  // namespace publisher

class TrajectoryViewer {
 public:
  template <int _N>
  static void PublishIMUData(std::shared_ptr<Trajectory<_N>> trajectory,
                             const Eigen::aligned_vector<IMUData> &imu_data) {
    if (publisher::pub_imu_raw_array_.getNumSubscribers() == 0 &&
        publisher::pub_imu_est_array_.getNumSubscribers() == 0)
      return;

    clins::imu_array imu_array_raw;
    clins::imu_array imu_array_est;

    for (auto const &v : imu_data) {
      if (!trajectory->GetTrajQuality(v.timestamp)) {
        continue;
      }
      geometry_msgs::Vector3 gyro, accel;
      tf::vectorEigenToMsg(v.gyro, gyro);
      tf::vectorEigenToMsg(v.accel, accel);
      imu_array_raw.timestamps.push_back(v.timestamp);
      imu_array_raw.angular_velocities.push_back(gyro);
      imu_array_raw.linear_accelerations.push_back(accel);

      auto const param = trajectory->GetTrajParam();

      Eigen::Vector3d w_b =
          trajectory->rotVelBody(v.timestamp) + param->gyro_bias;
      Eigen::Vector3d a_w = trajectory->transAccelWorld(v.timestamp);
      SE3d pose = trajectory->pose(v.timestamp);
      Eigen::Vector3d a_b =
          pose.so3().inverse() * (a_w + param->gravity) + param->acce_bias;

      geometry_msgs::Vector3 gyro2, accel2;
      tf::vectorEigenToMsg(w_b, gyro2);
      tf::vectorEigenToMsg(a_b, accel2);
      imu_array_est.timestamps.push_back(v.timestamp);
      imu_array_est.angular_velocities.push_back(gyro2);
      imu_array_est.linear_accelerations.push_back(accel2);
    }
    imu_array_raw.header.stamp = ros::Time::now();
    imu_array_raw.header.frame_id = "/imu";

    imu_array_est.header = imu_array_raw.header;

    publisher::pub_imu_raw_array_.publish(imu_array_raw);
    publisher::pub_imu_est_array_.publish(imu_array_est);
  }

  template <int _N>
  static void PublishIMUData(
      std::shared_ptr<Trajectory<_N>> trajectory,
      const Eigen::aligned_vector<IMUData> &imu_data,
      const std::vector<std::pair<double, IMUBias>> &imu_bias,
      std::string cache_path = " ") {
    //    if (publisher::pub_imu_raw_array_.getNumSubscribers() == 0 &&
    //        publisher::pub_imu_est_array_.getNumSubscribers() == 0)
    //      return;

    clins::imu_array imu_array_raw;
    clins::imu_array imu_array_est;
    int imu_bias_index = 0;
    for (auto const &v : imu_data) {
      if (!trajectory->GetTrajQuality(v.timestamp)) {
        continue;
      }
      geometry_msgs::Vector3 gyro, accel;
      tf::vectorEigenToMsg(v.gyro, gyro);
      tf::vectorEigenToMsg(v.accel, accel);
      imu_array_raw.timestamps.push_back(v.timestamp);
      imu_array_raw.angular_velocities.push_back(gyro);
      imu_array_raw.linear_accelerations.push_back(accel);

      auto const param = trajectory->GetCalibParam();

      Eigen::Vector3d gyro_bias, accel_bias;
      if (v.timestamp < imu_bias.front().first) {
        gyro_bias = Eigen::Vector3d(0, 0, 0);
        accel_bias = Eigen::Vector3d(0, 0, 0);
      } else if (v.timestamp > imu_bias.back().first) {
        gyro_bias = imu_bias.back().second.gyro_bias;
        accel_bias = imu_bias.back().second.accel_bias;
      } else {
        while (true) {
          if (v.timestamp >= imu_bias[imu_bias_index].first &&
              v.timestamp < imu_bias[imu_bias_index + 1].first) {
            gyro_bias = imu_bias[imu_bias_index].second.gyro_bias;
            accel_bias = imu_bias[imu_bias_index].second.accel_bias;
            break;
          }
          imu_bias_index++;
        }
      }

      Eigen::Vector3d w_b = trajectory->rotVelBody(v.timestamp) + gyro_bias;
      Eigen::Vector3d a_w = trajectory->transAccelWorld(v.timestamp);
      SE3d pose = trajectory->pose(v.timestamp);
      Eigen::Vector3d a_b =
          pose.so3().inverse() * (a_w + param->gravity) + accel_bias;

      geometry_msgs::Vector3 gyro2, accel2;
      tf::vectorEigenToMsg(w_b, gyro2);
      tf::vectorEigenToMsg(a_b, accel2);
      imu_array_est.timestamps.push_back(v.timestamp);
      imu_array_est.angular_velocities.push_back(gyro2);
      imu_array_est.linear_accelerations.push_back(accel2);
    }
    imu_array_raw.header.stamp = ros::Time::now();
    imu_array_raw.header.frame_id = "/imu";

    imu_array_est.header = imu_array_raw.header;

    publisher::pub_imu_raw_array_.publish(imu_array_raw);
    publisher::pub_imu_est_array_.publish(imu_array_est);

    if (cache_path != " ") {
      rosbag::Bag bag;
      bag.open(cache_path + "/plot_imu.bag", rosbag::bagmode::Write);
      bag.write("imu_raw_array", ros::Time::now(), imu_array_raw);
      bag.write("imu_est_array", ros::Time::now(), imu_array_est);
      bag.close();
    }
  }

  template <int _N>
  static void PublishViconData(
      std::shared_ptr<Trajectory<_N>> trajectory,
      const Eigen::aligned_vector<PoseData> &vicon_data) {
    if (publisher::pub_trajectory_raw_.getNumSubscribers() == 0 &&
        publisher::pub_trajectory_est_.getNumSubscribers() == 0)
      return;

    clins::pose_array vicon_path_raw;
    clins::pose_array vicon_path_est;

    for (auto const &v : vicon_data) {
      geometry_msgs::Vector3 position;
      geometry_msgs::Quaternion orientation;

      // raw data
      tf::vectorEigenToMsg(v.position, position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(), orientation);

      vicon_path_raw.timestamps.push_back(v.timestamp);
      vicon_path_raw.positions.push_back(position);
      vicon_path_raw.orientations.push_back(orientation);

      // estiamted pose
      SE3d pose;
      if (!trajectory->GetLidarPose(v.timestamp, pose)) continue;
      tf::vectorEigenToMsg(pose.translation(), position);
      tf::quaternionEigenToMsg(pose.unit_quaternion(), orientation);
      vicon_path_est.timestamps.push_back(v.timestamp);
      vicon_path_est.positions.push_back(position);
      vicon_path_est.orientations.push_back(orientation);
    }

    vicon_path_raw.header.frame_id = "/map";
    vicon_path_raw.header.stamp = ros::Time::now();
    vicon_path_est.header = vicon_path_raw.header;

    publisher::pub_trajectory_raw_.publish(vicon_path_raw);
    publisher::pub_trajectory_est_.publish(vicon_path_est);
  }

  static void PublishViconData(
      const Eigen::aligned_vector<PoseData> &vicon_est,
      const Eigen::aligned_vector<PoseData> &vicon_data) {
    if (publisher::pub_trajectory_raw_.getNumSubscribers() == 0 &&
        publisher::pub_trajectory_est_.getNumSubscribers() == 0)
      return;

    clins::pose_array vicon_path_raw;
    clins::pose_array vicon_path_est;

    for (auto const &v : vicon_data) {
      geometry_msgs::Vector3 position;
      geometry_msgs::Quaternion orientation;

      // raw data
      tf::vectorEigenToMsg(v.position, position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(), orientation);

      vicon_path_raw.timestamps.push_back(v.timestamp);
      vicon_path_raw.positions.push_back(position);
      vicon_path_raw.orientations.push_back(orientation);
    }

    for (auto const &v : vicon_est) {
      // estiamted pose
      geometry_msgs::Vector3 position;
      geometry_msgs::Quaternion orientation;

      tf::vectorEigenToMsg(v.position, position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(), orientation);

      vicon_path_est.timestamps.push_back(v.timestamp);
      vicon_path_est.positions.push_back(position);
      vicon_path_est.orientations.push_back(orientation);
    }

    vicon_path_raw.header.frame_id = "/map";
    vicon_path_raw.header.stamp = ros::Time::now();
    vicon_path_est.header = vicon_path_raw.header;

    publisher::pub_trajectory_raw_.publish(vicon_path_raw);
    publisher::pub_trajectory_est_.publish(vicon_path_est);
  }

  template <int _N>
  static void PublishIMUOrientationData(
      std::shared_ptr<Trajectory<_N>> trajectory,
      const Eigen::aligned_vector<PoseData> &orientation_data) {
    clins::pose_array imu_ori_path_raw;
    clins::pose_array imu_ori_path_est;

    for (auto const &v : orientation_data) {
      geometry_msgs::Vector3 position;
      geometry_msgs::Quaternion orientation;

      // raw data
      tf::vectorEigenToMsg(v.position, position);
      tf::quaternionEigenToMsg(v.orientation.unit_quaternion(), orientation);

      imu_ori_path_raw.timestamps.push_back(v.timestamp);
      imu_ori_path_raw.positions.push_back(position);
      imu_ori_path_raw.orientations.push_back(orientation);

      // estiamted pose
      SE3d pose = trajectory->pose(v.timestamp);
      tf::vectorEigenToMsg(pose.translation(), position);
      tf::quaternionEigenToMsg(pose.unit_quaternion(), orientation);
      imu_ori_path_est.timestamps.push_back(v.timestamp);
      imu_ori_path_est.positions.push_back(position);
      imu_ori_path_est.orientations.push_back(orientation);
    }

    imu_ori_path_raw.header.frame_id = "/map";
    imu_ori_path_raw.header.stamp = ros::Time::now();
    imu_ori_path_est.header = imu_ori_path_raw.header;

    publisher::pub_trajectory_raw_.publish(imu_ori_path_raw);
    publisher::pub_trajectory_est_.publish(imu_ori_path_est);
  }

  template <int _N>
  static void PublishSplineTrajectory(
      ros::Publisher &pub_trajectory,
      std::shared_ptr<Trajectory<_N>> trajectory, double min_time,
      double max_time, double dt) {
    if (min_time < trajectory->minTime()) min_time = trajectory->minTime();
    if (max_time > trajectory->maxTime()) max_time = trajectory->maxTime();

    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    std::vector<geometry_msgs::PoseStamped> poses_geo;
    for (double t = min_time; t < max_time; t += dt) {
      SE3d pose = trajectory->pose(t);
      geometry_msgs::PoseStamped poseIinG;
      poseIinG.header.stamp = t_temp.fromSec(t);
      poseIinG.header.frame_id = "/map";
      tf::pointEigenToMsg(pose.translation(), poseIinG.pose.position);
      tf::quaternionEigenToMsg(pose.unit_quaternion(),
                               poseIinG.pose.orientation);
      poses_geo.push_back(poseIinG);
    }

    nav_msgs::Path traj_path;
    traj_path.header.stamp = time_now;
    traj_path.header.frame_id = "/map";
    traj_path.poses = poses_geo;

    pub_trajectory.publish(traj_path);
  }

  template <int _N>
  static void PublishSplineTrajectory(
      std::shared_ptr<Trajectory<_N>> trajectory, double min_time,
      double max_time, double dt) {
    if (publisher::pub_spline_trajectory_.getNumSubscribers() == 0) return;

    if (min_time < trajectory->minTime()) min_time = trajectory->minTime();
    if (max_time > trajectory->maxTime()) max_time = trajectory->maxTime();

    ros::Time time_now = ros::Time::now();
    ros::Time t_temp;
    std::vector<geometry_msgs::PoseStamped> poses_geo;
    for (double t = min_time; t < max_time; t += dt) {
      SE3d pose = trajectory->pose(t);
      geometry_msgs::PoseStamped poseIinG;
      poseIinG.header.stamp = t_temp.fromSec(t);
      poseIinG.header.frame_id = "/map";
      tf::pointEigenToMsg(pose.translation(), poseIinG.pose.position);
      tf::quaternionEigenToMsg(pose.unit_quaternion(),
                               poseIinG.pose.orientation);
      poses_geo.push_back(poseIinG);
    }

    nav_msgs::Path traj_path;
    traj_path.header.stamp = time_now;
    traj_path.header.frame_id = "/map";
    traj_path.poses = poses_geo;

    publisher::pub_spline_trajectory_.publish(traj_path);
  }

  template <int _N>
  static void PublishDenseCloud(std::shared_ptr<Trajectory<_N>> trajectory,
                                LiDARFeature &target_feature,
                                LiDARFeature &source_feature) {
    static int skip = 0;
    if (publisher::pub_target_dense_cloud_.getNumSubscribers() != 0 &&
        (++skip % 20 == 0)) {
      PosCloud target_cloud;
      double target_timestamp = target_feature.timestamp;
      SE3d target_pos = trajectory->GetLidarPose(target_timestamp);
      Eigen::Matrix4d tranform_matrix = target_pos.matrix();
      pcl::transformPointCloud(*target_feature.surface_features, target_cloud,
                               tranform_matrix);

      sensor_msgs::PointCloud2 target_msg;
      pcl::toROSMsg(target_cloud, target_msg);
      target_msg.header.stamp = ros::Time::now();
      target_msg.header.frame_id = "/map";

      publisher::pub_target_dense_cloud_.publish(target_msg);
    }

    if (publisher::pub_source_dense_cloud_.getNumSubscribers() != 0) {
      VPointCloud source_cloud;
      for (size_t i = 0; i < source_feature.surface_features->size(); i++) {
        double point_timestamp =
            source_feature.surface_features->points[i].timestamp;
        SE3d point_pos = trajectory->GetLidarPose(point_timestamp);
        Eigen::Vector3d point_local(
            source_feature.surface_features->points[i].x,
            source_feature.surface_features->points[i].y,
            source_feature.surface_features->points[i].z);
        Eigen::Vector3d point_out =
            point_pos.so3() * point_local + point_pos.translation();
        VPoint p;
        p.x = point_out(0);
        p.y = point_out(1);
        p.z = point_out(2);
        source_cloud.push_back(p);
      }

      sensor_msgs::PointCloud2 source_msg;
      pcl::toROSMsg(source_cloud, source_msg);
      source_msg.header.stamp = ros::Time::now();
      source_msg.header.frame_id = "/map";
      publisher::pub_source_dense_cloud_.publish(source_msg);
    }
  }

  static void PublishICPCloud(GPointCloud::Ptr target_cloud,
                              GPointCloud::Ptr source_cloud,
                              Eigen::Matrix4f transform_matrix) {
    sensor_msgs::PointCloud2 target_msg;
    sensor_msgs::PointCloud2 source_msg;
    sensor_msgs::PointCloud2 raw_source_msg;
    pcl::toROSMsg(*target_cloud, target_msg);
    pcl::toROSMsg(*source_cloud, raw_source_msg);
    GPointCloud::Ptr transform_source_cloud(new GPointCloud);
    pcl::transformPointCloud(*source_cloud, *transform_source_cloud,
                             transform_matrix);
    pcl::toROSMsg(*transform_source_cloud, source_msg);

    target_msg.header.frame_id = "/map";
    source_msg.header.frame_id = "/map";
    raw_source_msg.header.frame_id = "/map";

    publisher::pub_icp_target_cloud_.publish(target_msg);
    publisher::pub_icp_source_cloud_.publish(source_msg);
    publisher::pub_icp_raw_source_cloud_.publish(raw_source_msg);
  }

  static void PublishICPCloud(GPointCloud::Ptr target_cloud,
                              GPointCloud::Ptr source_cloud,
                              Eigen::Matrix4f target_pose,
                              Eigen::Matrix4f transform_matrix) {
    sensor_msgs::PointCloud2 target_msg;
    sensor_msgs::PointCloud2 source_msg;

    GPointCloud target_cloud_in_G, source_cloud_in_G;
    pcl::transformPointCloud(*target_cloud, target_cloud_in_G, target_pose);
    pcl::transformPointCloud(*source_cloud, source_cloud_in_G,
                             target_pose * transform_matrix);

    pcl::toROSMsg(target_cloud_in_G, target_msg);
    pcl::toROSMsg(source_cloud_in_G, source_msg);

    target_msg.header.frame_id = "/map";
    source_msg.header.frame_id = "/map";

    publisher::pub_icp_target_cloud_.publish(target_msg);
    publisher::pub_icp_source_cloud_.publish(source_msg);
  }

  // 离散pose graph的显示
  static void PublishDiscretedPoseGraphMarker(std::vector<SE3d> pose_before,
                                              std::vector<SE3d> pose_after) {
    assert(pose_before.size() == pose_after.size() &&
           "[PublishDiscretedPoseGraphMarker] pose size error ");
    visualization_msgs::MarkerArray marker_array;
    // 优化前的节点
    visualization_msgs::Marker marker_node_before;
    marker_node_before.header.frame_id = "/map";
    marker_node_before.header.stamp = ros::Time::now();
    marker_node_before.action = visualization_msgs::Marker::ADD;
    marker_node_before.type = visualization_msgs::Marker::SPHERE_LIST;
    marker_node_before.ns = "pose_graph_before";
    marker_node_before.id = 0;
    marker_node_before.pose.orientation.w = 1;
    marker_node_before.scale.x = 0.05;
    marker_node_before.scale.y = 0.05;
    marker_node_before.scale.z = 0.05;
    marker_node_before.color.r = 1.0;
    marker_node_before.color.g = 0;
    marker_node_before.color.b = 0;
    marker_node_before.color.a = 1;

    // 优化后的节点
    visualization_msgs::Marker marker_node_after;
    marker_node_after.header.frame_id = "/map";
    marker_node_after.header.stamp = ros::Time::now();
    marker_node_after.action = visualization_msgs::Marker::ADD;
    marker_node_after.type = visualization_msgs::Marker::SPHERE_LIST;
    marker_node_after.ns = "pose_graph_after";
    marker_node_after.id = 0;
    marker_node_after.pose.orientation.w = 1;
    marker_node_after.scale.x = 0.05;
    marker_node_after.scale.y = 0.05;
    marker_node_after.scale.z = 0.05;
    marker_node_after.color.r = 0;
    marker_node_after.color.g = 0;
    marker_node_after.color.b = 1;
    marker_node_after.color.a = 1;

    visualization_msgs::Marker marker_edge;
    marker_edge.header.frame_id = "/map";
    marker_edge.header.stamp = ros::Time::now();
    marker_edge.action = visualization_msgs::Marker::ADD;
    marker_edge.type = visualization_msgs::Marker::LINE_LIST;
    marker_edge.ns = "loop_edges";
    marker_edge.id = 1;
    marker_edge.pose.orientation.w = 1;
    marker_edge.scale.x = 0.02;
    marker_edge.color.r = 0;
    marker_edge.color.g = 1.0;
    marker_edge.color.b = 0;
    marker_edge.color.a = 1;

    for (size_t i = 0; i < pose_after.size(); i++) {
      Eigen::Vector3d translation_before = pose_before[i].translation();
      geometry_msgs::Point p;
      p.x = translation_before(0);
      p.y = translation_before(1);
      p.z = translation_before(2);
      marker_node_before.points.push_back(p);
      marker_edge.points.push_back(p);

      Eigen::Vector3d translation_after = pose_after[i].translation();
      p.x = translation_after(0);
      p.y = translation_after(1);
      p.z = translation_after(2);
      marker_node_after.points.push_back(p);
      marker_edge.points.push_back(p);
    }

    marker_array.markers.push_back(marker_node_before);
    marker_array.markers.push_back(marker_node_after);
    marker_array.markers.push_back(marker_edge);

    publisher::pub_pose_graph_marker_.publish(marker_array);
  }
};

}  // namespace clins

#endif
