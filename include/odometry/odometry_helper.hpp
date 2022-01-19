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

#ifndef ODOMETRY_HELPER_HPP
#define ODOMETRY_HELPER_HPP

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_data/imu_data.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_broadcaster.h>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <clins/feature_cloud.h>
#include <feature/feature_extraction.h>
#include <odometry/imu_state_estimator.h>
#include <odometry/inertial_initializer.h>
#include <utils/gps_convert_utils.h>
#include <utils/vlp_common.h>
#include <visualization_msgs/MarkerArray.h>
#include <odometry/lidar_odometry.hpp>
#include <odometry/odom_visualizer.hpp>

namespace clins {

template <int _N>
class OdometryHelper {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<OdometryHelper<_N>> Ptr;

  OdometryHelper(const YAML::Node& node);

  void LidarSpinOffline();

 private:
  bool CreateCacheFolder(const std::string bag_path);

  void LiDARHandler(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg);

  void LiDARHandler(const velodyne_msgs::VelodyneScan::ConstPtr& lidar_msg);

  bool ParsePointCloud(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg,
                       double& max_time);

  bool ParsePointCloud(const velodyne_msgs::VelodyneScan::ConstPtr& lidar_msg,
                       double& max_time);

  void IMUHandler(const sensor_msgs::Imu::ConstPtr& imu_msg);

  void LoopClosure();

  bool TryToInitialize(double feature_time);

  void PublishKeyPoses();

  void PublishTF(Eigen::Quaterniond quat, Eigen::Vector3d pos,
                 std::string from_frame, std::string to_frame);

  void PublishLoopClosureMarkers();

  bool CheckMsgFields(const sensor_msgs::PointCloud2& cloud_msg,
                      std::string fields_name = "ring");

  ros::NodeHandle nh_;

  double knot_distance_;

  std::string bag_path_;
  std::string cache_path_;

  double bag_start_;
  double bag_durr_;

  std::string imu_topic_;
  std::string lidar_topic_;

  bool is_initialized_;

  bool save_map_;

  // loop closure
  bool loop_closure_enable_flag_;
  double loop_closure_frequency_;
  double loop_closure_timestamp_;

  ros::Subscriber sub_imu_;
  ros::Subscriber sub_lidar_feature_;
  ros::Subscriber sub_lidar_;

  ros::Publisher pub_key_pose_;
  ros::Publisher pub_loop_closure_marker_;

  CalibParamManager::Ptr calib_param_;
  typename Trajectory<_N>::Ptr trajectory_;
  typename LidarOdometry<_N>::Ptr lidar_odom_;
  typename TrajectoryManager<_N>::Ptr trajectory_manager_;
  std::shared_ptr<FeatureExtraction> feature_extraction_;

  std::shared_ptr<OdomInitializer> initializer_;
  std::shared_ptr<ImuStateEstimator> imu_state_estimator_;

  // raw lidar cloud
  RTPointCloud::Ptr raw_cloud_;
  // undistored lidar cloud
  RTPointCloud::Ptr undistort_cloud_;

  bool use_imu_orientation_;

  std::vector<double> lidar_timestamps_;
};

template <int _N>
OdometryHelper<_N>::OdometryHelper(const YAML::Node& node)
    : is_initialized_(false),
      loop_closure_timestamp_(-1),
      save_map_(false),
      raw_cloud_(new RTPointCloud),
      undistort_cloud_(new RTPointCloud) {
  /// load configuration from yaml
  knot_distance_ = node["knot_distance"].as<double>();
  imu_topic_ = node["imu_topic"].as<std::string>();
  lidar_topic_ = node["lidar_topic"].as<std::string>();

  loop_closure_enable_flag_ = node["loop_closure_enable_flag"].as<bool>();
  loop_closure_frequency_ = node["loop_closure_frequency"].as<double>();
  save_map_ = node["save_map"].as<bool>();

  bool use_corner_feature = node["use_corner_feature"].as<bool>();
  bool use_imu_orientation_ = node["use_imu_orientation"].as<bool>();

  calib_param_ = std::make_shared<CalibParamManager>(node);
  trajectory_ = std::make_shared<Trajectory<_N>>(knot_distance_);
  trajectory_->SetCalibParam(calib_param_);

  imu_state_estimator_ = std::make_shared<ImuStateEstimator>(node);
  trajectory_manager_ = std::make_shared<TrajectoryManager<_N>>(
      trajectory_, calib_param_, imu_state_estimator_);

  trajectory_manager_->SetUseCornerFeature(use_corner_feature);
  trajectory_manager_->SetUseIMUOrientation(use_imu_orientation_);

  lidar_odom_ = std::make_shared<LidarOdometry<_N>>(node, trajectory_);
  lidar_odom_->SetTrajectoryManager(trajectory_manager_);

  initializer_ = std::make_shared<OdomInitializer>(node);

  feature_extraction_ = std::make_shared<FeatureExtraction>(node);

  pub_key_pose_ = nh_.advertise<sensor_msgs::PointCloud2>("key_poses", 10);
  pub_loop_closure_marker_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "/loop_clousre_markers", 10);

  bag_path_ = node["bag_path"].as<std::string>();
  bag_start_ = node["bag_start"].as<double>();
  bag_durr_ = node["bag_durr"].as<double>();

  CreateCacheFolder(bag_path_);

  lidar_odom_->SetCachePath(cache_path_);
}

template <int _N>
bool OdometryHelper<_N>::CreateCacheFolder(const std::string path) {
  boost::filesystem::path p(path);
  if (p.extension() != ".bag") {
    return false;
  }
  cache_path_ = p.parent_path().string() + "/" + p.stem().string();

  boost::filesystem::create_directory(cache_path_);
  return true;
}

template <int _N>
void OdometryHelper<_N>::LiDARHandler(
    const sensor_msgs::PointCloud2::ConstPtr& lidar_msg) {
  double scan_time =
      lidar_msg->header.stamp.toSec() + calib_param_->time_offset;
  if (!is_initialized_) {
    if (TryToInitialize(scan_time)) {
      is_initialized_ = true;
    } else
      return;
  }

  if (scan_time < trajectory_->GetDataStartTime() ||
      trajectory_->GetDataStartTime() == 0) {
    std::cout << YELLOW << "skip scan : " << scan_time << RESET << std::endl;
    return;
  }

  // step1: Check msg
  double scan_max_time = 0;
  if (!ParsePointCloud(lidar_msg, scan_max_time)) return;
  scan_time -= trajectory_->GetDataStartTime();
  scan_max_time -= trajectory_->GetDataStartTime();
  lidar_timestamps_.push_back(scan_time);

  // step2: Integrate IMU measurements to initialize trajectoriy
  lidar_odom_->EstimateIMUMeasurement(scan_time, scan_max_time);

  // step3: Undistort Scan
  undistort_cloud_->clear();
  trajectory_->UndistortScan(scan_time, *raw_cloud_, scan_time,
                             *undistort_cloud_);

  // step4: Extract lidar feature
  feature_extraction_->LidarHandler(raw_cloud_, undistort_cloud_);

  lidar_odom_->FeatureCloudHandler(
      scan_time, scan_max_time, feature_extraction_->get_raw_corner_features(),
      feature_extraction_->get_raw_surface_features(), raw_cloud_);

  // step5: Update trajectory
  lidar_odom_->UpdateOdometry();
}

template <int _N>
void OdometryHelper<_N>::LiDARHandler(
    const velodyne_msgs::VelodyneScan::ConstPtr& lidar_msg) {
  double scan_time =
      lidar_msg->header.stamp.toSec() + calib_param_->time_offset;

  if (!is_initialized_) {
    if (TryToInitialize(scan_time)) {
      is_initialized_ = true;
    } else
      return;
  }

  if (scan_time < trajectory_->GetDataStartTime() ||
      trajectory_->GetDataStartTime() == 0) {
    std::cout << YELLOW << "skip scan : " << scan_time << RESET << std::endl;
    return;
  }

  // step1: Check msg
  double scan_max_time = 0;
  if (!ParsePointCloud(lidar_msg, scan_max_time)) return;
  scan_time -= trajectory_->GetDataStartTime();
  scan_max_time -= trajectory_->GetDataStartTime();
  lidar_timestamps_.push_back(scan_time);

  // step2: Integrate IMU measurements to initialize trajectoriy
  lidar_odom_->EstimateIMUMeasurement(scan_time, scan_max_time);

  // step3: Undistort Scan
  undistort_cloud_->clear();
  trajectory_->UndistortScan(scan_time, *raw_cloud_, scan_time,
                             *undistort_cloud_);

  // step4: Extract lidar feature
  feature_extraction_->LidarHandler(raw_cloud_, undistort_cloud_);

  lidar_odom_->FeatureCloudHandler(
      scan_time, scan_max_time, feature_extraction_->get_raw_corner_features(),
      feature_extraction_->get_raw_surface_features(), raw_cloud_);

  // step5: Update trajectory
  lidar_odom_->UpdateOdometry();
}

template <int _N>
bool OdometryHelper<_N>::ParsePointCloud(
    const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, double& max_time) {
  bool hasTimeField = false;
  bool check_field = true;
  if (check_field) {
    bool ret = CheckMsgFields(*lidar_msg, "ring");
    ret = ret && CheckMsgFields(*lidar_msg, "time");
    check_field = false;

    if (ret) hasTimeField = true;
  }

  double scan_time =
      lidar_msg->header.stamp.toSec() + calib_param_->time_offset;

  max_time = 0;
  /// convert cloud
  if (hasTimeField) {
    RTPointCloud::Ptr cur_cloud(new RTPointCloud());
    pcl::fromROSMsg(*lidar_msg, *cur_cloud);
    raw_cloud_->clear();
    for (size_t i = 0; i < cur_cloud->size(); i++) {
      double point_timestamp = scan_time + cur_cloud->points[i].time;
      raw_cloud_->push_back(cur_cloud->points[i]);
      if (max_time < point_timestamp) max_time = point_timestamp;
    }
    return true;
  } else {
    return false;
  }
}

template <int _N>
bool OdometryHelper<_N>::ParsePointCloud(
    const velodyne_msgs::VelodyneScan::ConstPtr& lidar_msg, double& max_time) {
  static VelodyneCorrection vc;
  RTPointCloud::Ptr cur_cloud(new RTPointCloud());
  vc.unpack_scan(lidar_msg, *cur_cloud);
  raw_cloud_->clear();

  double scan_time =
      lidar_msg->header.stamp.toSec() + calib_param_->time_offset;
  ;
  for (size_t i = 0; i < cur_cloud->size(); i++) {
    double point_timestamp = scan_time + cur_cloud->points[i].time;
    raw_cloud_->push_back(cur_cloud->points[i]);
    if (max_time < point_timestamp) max_time = point_timestamp;
  }
  return true;
}

template <int _N>
void OdometryHelper<_N>::IMUHandler(const sensor_msgs::Imu::ConstPtr& imu_msg) {
  IMUData data;
  data.timestamp = imu_msg->header.stamp.toSec();
  data.gyro =
      Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
                      imu_msg->angular_velocity.z);
  data.accel = Eigen::Vector3d(imu_msg->linear_acceleration.x,
                               imu_msg->linear_acceleration.y,
                               imu_msg->linear_acceleration.z);
  if (use_imu_orientation_) {
    data.orientation = SO3d(
        Eigen::Quaterniond(imu_msg->orientation.w, imu_msg->orientation.x,
                           imu_msg->orientation.y, imu_msg->orientation.z));
  }
  if (!is_initialized_) {
    initializer_->FeedIMUData(data);
  }
  trajectory_manager_->AddIMUData(data);
}

template <int _N>
void OdometryHelper<_N>::LoopClosure() {
  if (!loop_closure_enable_flag_) return;
  lidar_odom_->LoopClosureHandler();
  PublishLoopClosureMarkers();
}

template <int _N>
void OdometryHelper<_N>::LidarSpinOffline() {
  rosbag::Bag bag;
  bag.open(bag_path_, rosbag::bagmode::Read);

  std::vector<std::string> topics;
  topics.push_back(lidar_topic_);
  topics.push_back(imu_topic_);

  rosbag::View view_;
  rosbag::View view_full;
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  time_init += ros::Duration(bag_start_);
  ros::Time time_finish = (bag_durr_ < 0)
                              ? view_full.getEndTime()
                              : time_init + ros::Duration(bag_durr_);
  view_.addQuery(bag, rosbag::TopicQuery(topics), time_init, time_finish);

  if (view_.size() == 0) {
    ROS_ERROR("No messages to play on specified topics.  Exiting.");
    ros::shutdown();
    return;
  }

  ros::Rate rate(5);
  for (const rosbag::MessageInstance& m : view_) {
    ros::Time ros_bag_time = m.getTime();
    if (m.getTopic() == imu_topic_) {
      sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
      IMUHandler(imu_msg);
    } else if (m.getTopic() == lidar_topic_) {
      if (m.getDataType() == std::string("sensor_msgs/PointCloud2")) {
        auto lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
        double delta_time =
            ros_bag_time.toSec() - lidar_msg->header.stamp.toSec();
        if (delta_time < 0.08) {
          std::cout << " Delta Time : " << delta_time << std::endl;
        }
        LiDARHandler(lidar_msg);
      } else if (m.getDataType() == std::string("velodyne_msgs/VelodyneScan")) {
        auto lidar_msg = m.instantiate<velodyne_msgs::VelodyneScan>();
        double delta_time =
            ros_bag_time.toSec() - lidar_msg->header.stamp.toSec();
        if (delta_time < 0.08) {
          std::cout << " Delta Time : " << delta_time << std::endl;
        }
        LiDARHandler(lidar_msg);
      }

      if (is_initialized_) {
        PublishKeyPoses();

        auto pose = trajectory_->GetLidarPose(lidar_timestamps_.back());
        PublishTF(pose.unit_quaternion(), pose.translation(), "lidar", "map");
      }
    }

    double bag_time = m.getTime().toSec();
    if (loop_closure_timestamp_ < 0) {
      loop_closure_timestamp_ = bag_time;
    } else {
      if (bag_time - loop_closure_timestamp_ >
          (1.0 / loop_closure_frequency_)) {
        loop_closure_timestamp_ = bag_time;
        LoopClosure();
      }
    }

    if (!ros::ok()) break;
  }

  // save map
  if (save_map_) {
    std::cout << GREEN << "\n\n";

    lidar_odom_->SaveKeyFrameCloud(cache_path_);
    trajectory_->SaveTrajectoryControlPoints(cache_path_ +
                                             "/trajectory_control_points.txt");
    trajectory_->TrajectoryToTUMTxt(cache_path_, bag_start_,
                                    bag_start_ + bag_durr_, 0, 0.05);
    trajectory_->LidarTrajectoryToTUMTxt(cache_path_, bag_start_,
                                         bag_start_ + bag_durr_, 0, 0.05);

    std::cout << RESET << "\n\n";
  }
}

template <int _N>
bool OdometryHelper<_N>::TryToInitialize(double feature_time) {
  bool success = initializer_->IMUInitializer();
  if (!success) return false;

  calib_param_->SetGravity(initializer_->GetGravity());
  calib_param_->SetAccelBias(initializer_->GetAccelBias());
  calib_param_->SetGyroBias(initializer_->GetGyroBias());

  trajectory_manager_->InitIMUData(feature_time);

  trajectory_manager_->SetInitialPoseRotation(initializer_->GetI0ToG());

  return true;
}

template <int _N>
void OdometryHelper<_N>::PublishKeyPoses() {
  PosCloud::Ptr key_poses = lidar_odom_->GetCloudKeyPose();
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*key_poses, cloud_msg);
  cloud_msg.header.frame_id = "map";
  pub_key_pose_.publish(cloud_msg);
}

template <int _N>
void OdometryHelper<_N>::PublishTF(Eigen::Quaterniond quat, Eigen::Vector3d pos,
                                   std::string from_frame,
                                   std::string to_frame) {
  static tf::TransformBroadcaster tbr;
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(pos[0], pos[1], pos[2]));
  tf::Quaternion tf_q(quat.x(), quat.y(), quat.z(), quat.w());
  transform.setRotation(tf_q);
  tbr.sendTransform(
      tf::StampedTransform(transform, ros::Time::now(), to_frame, from_frame));
}

template <int _N>
void OdometryHelper<_N>::PublishLoopClosureMarkers() {
  std::map<int, int> loop_closure_info =
      lidar_odom_->GetHistoryLoopClosureInfo();

  PosCloud::Ptr key_poses = lidar_odom_->GetCloudKeyPose();

  if (loop_closure_info.empty()) return;
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker_node;
  marker_node.header.frame_id = "/map";
  marker_node.header.stamp = ros::Time::now();
  marker_node.action = visualization_msgs::Marker::ADD;
  marker_node.type = visualization_msgs::Marker::SPHERE_LIST;
  marker_node.ns = "loop_nodes";
  marker_node.id = 0;
  marker_node.pose.orientation.w = 1;
  marker_node.scale.x = 0.3;
  marker_node.scale.y = 0.3;
  marker_node.scale.z = 0.3;
  marker_node.color.r = 0;
  marker_node.color.g = 0.8;
  marker_node.color.b = 1;
  marker_node.color.a = 1;

  // loop edges
  visualization_msgs::Marker marker_edge;
  marker_edge.header.frame_id = "/map";
  marker_edge.header.stamp = ros::Time::now();
  marker_edge.action = visualization_msgs::Marker::ADD;
  marker_edge.type = visualization_msgs::Marker::LINE_LIST;
  marker_edge.ns = "loop_edges";
  marker_edge.id = 1;
  marker_edge.pose.orientation.w = 1;
  marker_edge.scale.x = 0.1;
  marker_edge.color.r = 0.9;
  marker_edge.color.g = 0.9;
  marker_edge.color.b = 0;
  marker_edge.color.a = 1;

  for (auto iter = loop_closure_info.begin(); iter != loop_closure_info.end();
       iter++) {
    int key_cur = iter->first;
    int key_history = iter->second;
    geometry_msgs::Point p;
    p.x = key_poses->points[key_cur].x;
    p.y = key_poses->points[key_cur].y;
    p.z = key_poses->points[key_cur].z;
    marker_node.points.push_back(p);
    marker_edge.points.push_back(p);
    p.x = key_poses->points[key_history].x;
    p.y = key_poses->points[key_history].y;
    p.z = key_poses->points[key_history].z;
    marker_node.points.push_back(p);
    marker_edge.points.push_back(p);
  }

  marker_array.markers.push_back(marker_node);
  marker_array.markers.push_back(marker_edge);
  pub_loop_closure_marker_.publish(marker_array);
}

template <int _N>
bool OdometryHelper<_N>::CheckMsgFields(
    const sensor_msgs::PointCloud2& cloud_msg, std::string fields_name) {
  bool flag = false;
  for (int i = 0; i < cloud_msg.fields.size(); ++i) {
    if (cloud_msg.fields[i].name == fields_name) {
      flag = true;
    }
    if (flag) break;
  }
  return flag;
}

}  // namespace clins

#endif
