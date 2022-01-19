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
#include <ros/package.h>
#include <ros/ros.h>

using namespace std;
using namespace clins;

int main(int argc, char** argv) {
  ros::init(argc, argv, "feature_extration_node");
  ros::NodeHandle nh("~");

  std::string config_path;
  nh.param<std::string>("config_path", config_path, "/config/ct_odometry.yaml");

  std::string package_name = "clins";
  std::string package_path = ros::package::getPath(package_name);

  std::string config_file_path = package_path + config_path;
  YAML::Node config_node = YAML::LoadFile(config_file_path);

  FeatureExtraction FE(config_node);

  ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");

  ros::spin();

  return 0;
}
