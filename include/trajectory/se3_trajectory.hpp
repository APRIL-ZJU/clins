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

#ifndef SE3_TRAJECTORY_HPP
#define SE3_TRAJECTORY_HPP

#include <basalt/spline/se3_spline.h>
#include <sensor_data/lidar_data.h>
#include <sensor_data/calibration.hpp>
#include <sophus/se3.hpp>
#include <string>

namespace clins {

template <int _N>
class Trajectory : public basalt::Se3Spline<_N, double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<Trajectory<_N>> Ptr;

  Trajectory(double time_interval, double start_time = 0, size_t segment_id = 0)
      : basalt::Se3Spline<_N, double>(time_interval, start_time),
        active_time_(start_time),
        data_start_time_(start_time),
        forced_fixed_time_(-1) {
    this->extendKnotsTo(start_time, SO3d(Eigen::Quaterniond::Identity()),
                        Eigen::Vector3d(0, 0, 0));
  }

  void SetCalibParam(std::shared_ptr<CalibParamManager> calib_param) {
    calib_param_ = calib_param;
  }

  const std::shared_ptr<CalibParamManager> GetCalibParam() const {
    return calib_param_;
  }

  /// The control points before active_time are ready for query
  void SetActiveTime(double active_time) { active_time_ = active_time; }

  void UpdateActiveTime(double measurement_time) {
    size_t knot_index = this->computeTIndex(measurement_time).second;
    double time = ((int)knot_index - 3) * this->getDt();
    active_time_ = std::max(time, active_time_);
    //    std::cout << RED << "[ActiveTime]  :  " << active_time_ << RESET
    //              << std::endl;
  }

  double GetActiveTime() const { return active_time_; }

  void SetForcedFixedTime(double time) {
    forced_fixed_time_ = std::max(forced_fixed_time_, time);
  }

  double GetForcedFixedTime() const { return forced_fixed_time_; }

  bool GetTrajQuality(const double timestamp) const {
    if (timestamp < this->minTime() || timestamp >= this->maxTime())
      return false;
    else
      return true;
  }

  /// The start timestamp of bag
  void setDataStartTime(double data_start_time) {
    data_start_time_ = data_start_time;
  }

  double GetDataStartTime() const { return data_start_time_; }

  SE3d GetLidarPose(const double timestamp) const {
    double t = timestamp + calib_param_->time_offset;
    if (t < this->minTime()) {
      t = this->minTime();
    } else if (t >= this->maxTime()) {
      t = this->maxTime() - 1e-9;
    }

    SE3d pose_I_to_G = this->pose(t);
    SE3d pose_L_to_G = pose_I_to_G * calib_param_->se3_LtoI;
    return pose_L_to_G;
  }

  bool GetLidarPose(const double timestamp, SE3d &lidar_pose) {
    double t = timestamp + calib_param_->time_offset;
    if (t < this->minTime() || t >= this->maxTime()) return false;

    SE3d pose_I_to_G = this->pose(t);
    SE3d pose_L_to_G = pose_I_to_G * calib_param_->se3_LtoI;
    lidar_pose = pose_L_to_G;
    return true;
  }

  void UndistortScan(const PosCloud &scan_raw, const double target_timestamp,
                     PosCloud &scan_in_target) const;

  void UndistortScan(double scan_timestamp, const RTPointCloud &scan_raw,
                     const double target_timestamp,
                     RTPointCloud &scan_in_target) const;

  void SaveTrajectoryControlPoints(std::string path);

  bool LoadTrajectoryControlPoints(std::string path);

  void TrajectoryToTUMTxt(std::string file_path, double relative_start_time = 0,
                          double relative_end_time = 0, int iteration_num = 0,
                          double dt = 0.02);

  void LidarTrajectoryToTUMTxt(std::string file_path,
                               double relative_start_time = 0,
                               double relative_end_time = 0,
                               int iteration_num = 0, double dt = 0.02);

  CalibParamManager::Ptr calib_param_;

 private:
  double active_time_;
  double data_start_time_;
  // The control points less than this time are fixed during opt
  double forced_fixed_time_;
};

template <int _N>
void Trajectory<_N>::UndistortScan(const PosCloud &scan_raw,
                                   const double target_timestamp,
                                   PosCloud &scan_in_target) const {
  scan_in_target.header = scan_raw.header;
  scan_in_target.resize(scan_raw.size());
  scan_in_target.is_dense = true;

  SE3d pose_G_to_target = GetLidarPose(target_timestamp).inverse();

  std::size_t cnt = 0;
  for (auto const &raw_p : scan_raw.points) {
    if (pcl_isnan(raw_p.x)) {
      scan_in_target.is_dense = false;
      std::cout << RED << "[UndistortScan] input cloud exists NAN point\n"
                << RESET;
      continue;
    }
    SE3d pose_Lk_to_G = GetLidarPose(raw_p.timestamp);

    Eigen::Vector3d p_Lk(raw_p.x, raw_p.y, raw_p.z);
    Eigen::Vector3d point_out;
    point_out = pose_G_to_target * pose_Lk_to_G * p_Lk;

    PosPoint point;
    point.x = point_out(0);
    point.y = point_out(1);
    point.z = point_out(2);
    point.timestamp = raw_p.timestamp;

    scan_in_target[cnt++] = point;
  }
}

template <int _N>
void Trajectory<_N>::UndistortScan(double scan_timestamp,
                                   const RTPointCloud &scan_raw,
                                   const double target_timestamp,
                                   RTPointCloud &scan_in_target) const {
  scan_in_target.header = scan_raw.header;
  scan_in_target.resize(scan_raw.size());
  scan_in_target.is_dense = true;

  SE3d pose_G_to_target = GetLidarPose(target_timestamp).inverse();

  std::size_t cnt = 0;
  for (auto const &raw_p : scan_raw.points) {
    if (pcl_isnan(raw_p.x)) {
      scan_in_target.is_dense = false;
      std::cout << RED << "[UndistortScan] input cloud exists NAN point\n"
                << RESET;
      continue;
    }
    SE3d pose_Lk_to_G = GetLidarPose(scan_timestamp + raw_p.time);

    Eigen::Vector3d p_Lk(raw_p.x, raw_p.y, raw_p.z);
    Eigen::Vector3d point_out;
    point_out = pose_G_to_target * pose_Lk_to_G * p_Lk;

    RTPoint point;
    point.x = point_out(0);
    point.y = point_out(1);
    point.z = point_out(2);
    point.intensity = raw_p.intensity;
    point.ring = raw_p.ring;
    point.time = raw_p.time;

    scan_in_target[cnt++] = point;
  }
}

template <int _N>
void Trajectory<_N>::SaveTrajectoryControlPoints(std::string path) {
  std::ofstream outfile;
  outfile.open(path);
  size_t NumKnots = this->numKnots();

  outfile.precision(20);
  outfile << this->data_start_time_ << " " << this->getDt() << " " << NumKnots
          << "\n";

  outfile.precision(6);
  for (size_t i = 0; i < NumKnots; i++) {
    Eigen::Vector3d p = this->getKnotPos(i);
    Sophus::SO3d s = this->getKnotSO3(i);
    Eigen::Quaterniond q = s.unit_quaternion();
    outfile << p(0) << " " << p(1) << " " << p(2) << " " << q.x() << " "
            << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
  outfile.close();
}

template <int _N>
bool Trajectory<_N>::LoadTrajectoryControlPoints(std::string path) {
  std::ifstream infile;
  infile.open(path);
  std::string current_line;

  std::getline(infile, current_line);
  {
    std::istringstream s(current_line);  // Loop variables
    std::string field;
    std::vector<double> vec;
    // Loop through this line
    while (std::getline(s, field, ' ')) {
      if (field.empty())  // Skip if empty
        continue;
      vec.push_back(std::atof(field.c_str()));  // save the data to our vector
    }
    assert(vec.size() == 3 && "The second line has wrong size.. expected 3");
    double start_time = vec.at(0);
    this->data_start_time_ = start_time;
  }

  std::vector<Eigen::VectorXd> controlPoints;
  while (std::getline(infile, current_line)) {
    std::istringstream s(current_line);
    std::string field;
    std::vector<double> vec;

    while (std::getline(s, field, ' ')) {
      if (field.empty())  // Skip if empty
        continue;
      // save the data to our vector
      vec.push_back(std::atof(field.c_str()));
    }
    // Create eigen vector
    Eigen::VectorXd temp(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
      temp(i) = vec.at(i);
    }
    controlPoints.push_back(temp);
  }

  std::cout << "control point size : " << controlPoints.size() << std::endl;

  for (unsigned int i = 0; i < controlPoints.size(); i++) {
    Eigen::VectorXd temp = controlPoints.at(i);
    Eigen::Vector3d p = temp.head<3>();
    Eigen::Quaterniond q(temp(6), temp(3), temp(4), temp(5));
    Sophus::SE3d se3_knot(q, p);
    if (i < this->numKnots()) {
      this->setKnot(se3_knot, i);
    } else
      this->knots_push_back(se3_knot);
  }

  return true;
}

template <int _N>
void Trajectory<_N>::TrajectoryToTUMTxt(std::string file_path,
                                        double relative_start_time,
                                        double relative_end_time,
                                        int iteration_num, double dt) {
  std::string file_name = "/trajectory-" + std::to_string(relative_start_time) +
                          "-" + std::to_string(relative_end_time) + "-iter" +
                          std::to_string(iteration_num) + ".txt";
  std::string traj_path = file_path + file_name;

  std::ofstream outfile;
  outfile.open(traj_path);

  double min_time = this->minTime();
  double max_time = this->maxTime();
  for (double t = min_time; t < max_time; t += dt) {
    double relative_bag_time = relative_start_time + t;

    SE3d pose = this->pose(t);
    Eigen::Vector3d p = pose.translation();
    Eigen::Quaterniond q = pose.unit_quaternion();

    outfile.precision(9);
    outfile << relative_bag_time << " ";
    outfile.precision(5);
    outfile << p(0) << " " << p(1) << " " << p(2) << " " << q.x() << " "
            << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
  outfile.close();
  std::cout << "Save trajectory at " << traj_path << std::endl;
}

template <int _N>
void Trajectory<_N>::LidarTrajectoryToTUMTxt(std::string file_path,
                                             double relative_start_time,
                                             double relative_end_time,
                                             int iteration_num, double dt) {
  std::string file_name = "/lidar_trajectory-" +
                          std::to_string(relative_start_time) + "-" +
                          std::to_string(relative_end_time) + "-iter" +
                          std::to_string(iteration_num) + ".txt";
  std::string traj_path = file_path + file_name;

  std::ofstream outfile;
  outfile.open(traj_path);

  double min_time = this->minTime();
  double max_time = this->maxTime();
  for (double t = min_time; t < max_time; t += dt) {
    double relative_bag_time = relative_start_time + t;

    SE3d pose = this->GetLidarPose(t);
    Eigen::Vector3d p = pose.translation();
    Eigen::Quaterniond q = pose.unit_quaternion();

    outfile.precision(9);
    outfile << relative_bag_time << " ";
    outfile.precision(5);
    outfile << p(0) << " " << p(1) << " " << p(2) << " " << q.x() << " "
            << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
  outfile.close();
  std::cout << "Save trajectory at " << traj_path << std::endl;
}

}  // namespace clins

#endif
