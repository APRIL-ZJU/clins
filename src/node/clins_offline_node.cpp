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

#include <ros/package.h>
#include <ros/ros.h>

#include <clins/feature_cloud.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <odometry/odometry_helper.hpp>
#include <trajectory/trajectory_viewer.hpp>

using namespace std;
using namespace clins;

int main(int argc, char** argv) {
  ros::init(argc, argv, "clins_offline_node");
  ros::NodeHandle nh("~");

  clins::publisher::SetPublisher(nh);

  std::string package_name = "clins";
  std::string package_path = ros::package::getPath(package_name);

  std::string config_name;
  nh.param<std::string>("config_name", config_name, "/config/ct_odometry.yaml");

  std::string config_file_path = package_path + config_name;
  YAML::Node config_node = YAML::LoadFile(config_file_path);

  OdometryHelper<4> lio(config_node);

  lio.LidarSpinOffline();

  return 0;
}
