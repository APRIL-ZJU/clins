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

#include <ros/ros.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include <trajectory/se3_trajectory.hpp>

using namespace std;
using namespace clins;

int main(int argc, char** argv) {
  ros::init(argc, argv, "kaist_sick_map_generation");
  ros::NodeHandle nh("~");

  double knot_distance;
  std::string trajectory_path;
  std::string bag_path;
  std::string save_cloud_path;

  nh.param<double>("knot_distance", knot_distance, 0.1);
  nh.param<std::string>("trajectory_path", trajectory_path,
                        "kaist-urban-07/trajectory_control_points.txt");
  nh.param<std::string>("bag_path", bag_path, "kaist-urban-07.bag");
  nh.param<std::string>("save_cloud_path", save_cloud_path, "sick_map.pcd");

  /// Step1 : Load trajectory
  Trajectory<4>::Ptr trajectory_(new Trajectory<4>(knot_distance));
  trajectory_->LoadTrajectoryControlPoints(trajectory_path);
  double trajectory_start_time = trajectory_->GetDataStartTime();
  double trajectory_end_time = trajectory_start_time + trajectory_->maxTime();

  /// Step2 : Calculate external parameters
  SE3d vechile2IMU(Eigen::Quaterniond::Identity(),
                   Eigen::Vector3d(-0.07, 0, 1.7));
  Eigen::Matrix3d rot_vechile2back_sick;
  rot_vechile2back_sick << -0.0100504, -0.70497, 0.709166, -0.999698,
      -0.00880889, -0.0229246, 0.0224082, -0.709182, -0.704669;
  Eigen::Quaterniond q_v2bs(rot_vechile2back_sick);
  SE3d vechile2BackSick(q_v2bs, Eigen::Vector3d(-0.660887, 0.0170107, 1.63466));

  Eigen::Matrix3d rot_vechile2middle_sick;
  rot_vechile2middle_sick << 0.00246613, 0.797555, 0.603241, 0.999991,
      5.11536e-05, -0.00415573, -0.00334528, 0.603246, -0.797548;
  Eigen::Quaterniond q_v2ms(rot_vechile2middle_sick);
  SE3d vechile2MiddleSick(q_v2ms,
                          Eigen::Vector3d(0.731734, -0.0281114, 1.78419));

  SE3d sick_back_in_imu = vechile2IMU.inverse() * vechile2BackSick;
  SE3d sick_middle_in_imu = vechile2IMU.inverse() * vechile2MiddleSick;

  /// Step3 : Load Sick data from bag
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);

  std::string back_sick_topic = "/sick_back_scan";
  std::string middle_sick_topic = "/sick_front_scan";
  std::vector<std::string> topics;
  topics.push_back(back_sick_topic);
  topics.push_back(middle_sick_topic);

  rosbag::View view_;
  rosbag::View view_full;
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  view_.addQuery(bag, rosbag::TopicQuery(topics), time_init,
                 view_full.getEndTime());

  std::vector<std::pair<double, RTPointCloud>> sick_back_cloud_cache;
  std::vector<std::pair<double, RTPointCloud>> sick_middle_cloud_cache;

  for (const rosbag::MessageInstance& m : view_) {
    if (m.getTopic() == back_sick_topic) {
      sensor_msgs::PointCloud2::ConstPtr sick_msg =
          m.instantiate<sensor_msgs::PointCloud2>();
      if (sick_msg != NULL) {
        double timestamp = sick_msg->header.stamp.toSec();
        if (timestamp < trajectory_start_time ||
            timestamp > trajectory_end_time)
          continue;
        timestamp -= trajectory_->GetDataStartTime();
        RTPointCloud cloud;
        pcl::fromROSMsg(*sick_msg, cloud);
        sick_back_cloud_cache.push_back(std::make_pair(timestamp, cloud));
      }
    } else if (m.getTopic() == middle_sick_topic) {
      sensor_msgs::PointCloud2::ConstPtr sick_msg =
          m.instantiate<sensor_msgs::PointCloud2>();
      if (sick_msg != NULL) {
        double timestamp = sick_msg->header.stamp.toSec();
        if (timestamp < trajectory_start_time ||
            timestamp > trajectory_end_time)
          continue;
        timestamp -= trajectory_->GetDataStartTime();
        RTPointCloud cloud;
        pcl::fromROSMsg(*sick_msg, cloud);
        sick_middle_cloud_cache.push_back(std::make_pair(timestamp, cloud));
      }
    }
  }

  /// Step4 : Merge map
  VPointCloud::Ptr full_map(new VPointCloud);
  for (size_t i = 0; i < sick_back_cloud_cache.size(); i++) {
    RTPointCloud lidar_cloud = sick_back_cloud_cache[i].second;
    double time_base = sick_back_cloud_cache[i].first;
    if (time_base >= trajectory_->maxTime()) break;
    VPointCloud local_cloud;
    for (size_t j = 0; j < lidar_cloud.points.size(); j += 1) {
      VPoint p;
      p.x = lidar_cloud.points[j].x;
      p.y = lidar_cloud.points[j].y;
      p.z = lidar_cloud.points[j].z;
      p.intensity = lidar_cloud.points[j].intensity;
      double distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      local_cloud.push_back(p);
    }

    SE3d imu_pose = trajectory_->pose(time_base);
    SE3d sick_back_pose = imu_pose * sick_back_in_imu;
    Eigen::Matrix4f sick_back_matrix = sick_back_pose.matrix().cast<float>();
    VPointCloud global_cloud;
    pcl::transformPointCloud(local_cloud, global_cloud, sick_back_matrix);
    *full_map += global_cloud;
  }

  for (size_t i = 0; i < sick_middle_cloud_cache.size(); i++) {
    RTPointCloud lidar_cloud = sick_middle_cloud_cache[i].second;
    double time_base = sick_middle_cloud_cache[i].first;
    if (time_base >= trajectory_->maxTime()) break;
    VPointCloud local_cloud;
    for (size_t j = 0; j < lidar_cloud.points.size(); j += 1) {
      VPoint p;
      p.x = lidar_cloud.points[j].x;
      p.y = lidar_cloud.points[j].y;
      p.z = lidar_cloud.points[j].z;
      p.intensity = lidar_cloud.points[j].intensity;
      local_cloud.push_back(p);
    }

    SE3d imu_pose = trajectory_->pose(time_base);
    SE3d sick_back_pose = imu_pose * sick_middle_in_imu;
    Eigen::Matrix4f sick_back_matrix = sick_back_pose.matrix().cast<float>();
    VPointCloud global_cloud;
    pcl::transformPointCloud(local_cloud, global_cloud, sick_back_matrix);
    *full_map += global_cloud;
  }

  pcl::io::savePCDFileBinaryCompressed(save_cloud_path, *full_map);

  return 0;
}
