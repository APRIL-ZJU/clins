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

#ifndef ODOM_VISUALIZER_HPP
#define ODOM_VISUALIZER_HPP

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_data/lidar_data.h>

#include <odometry/lidar_odometry.hpp>

namespace clins {

class CloudPublisher {
 public:
  CloudPublisher(ros::NodeHandle& nh, std::string topic_name, size_t buff_size,
                 std::string frame_id)
      : nh_(nh), frame_id_(frame_id) {
    publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(topic_name, buff_size);
  }

  void Publish(VPointCloud::Ptr cloud_ptr_input) {
    sensor_msgs::PointCloud2Ptr cloud_ptr_output(
        new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*cloud_ptr_input, *cloud_ptr_output);
    cloud_ptr_output->header.stamp = ros::Time::now();
    cloud_ptr_output->header.frame_id = frame_id_;
    publisher_.publish(*cloud_ptr_output);
  }

 private:
  ros::NodeHandle nh_;
  ros::Publisher publisher_;
  std::string frame_id_;
};

}  // namespace clins

#endif
