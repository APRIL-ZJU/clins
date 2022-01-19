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

#ifndef LIDAR_DATA_H
#define LIDAR_DATA_H

#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl_conversions/pcl_conversions.h>
#include <utils/eigen_utils.hpp>

// the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET "\033[0m"
#define BLACK "\033[30m"   /* Black */
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

typedef pcl::PointXYZI VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;

typedef pcl::PointXYZ GPoint;
typedef pcl::PointCloud<GPoint> GPointCloud;

/// reference
/// https://github.com/ros-drivers/velodyne/blob/master/velodyne_pcl/include/velodyne_pcl/point_types.h
namespace velodyne_pcl {
struct PointXYZIRT {
  PCL_ADD_POINT4D;                 // quad-word XYZ
  float intensity;                 ///< laser intensity reading
  uint16_t ring;                   ///< laser ring number
  float time;                      ///< laser time reading
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;

struct PointXYZT {
  PCL_ADD_POINT4D;                 /// quad-word XYZ
  double timestamp;                /// laser timestamp
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;

struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

}  // namespace velodyne_pcl

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_pcl::PointXYZIRT, (float, x, x)  //
                                  (float, y, y)                             //
                                  (float, z, z)                             //
                                  (float, intensity, intensity)             //
                                  (uint16_t, ring, ring)                    //
                                  (float, time, time)                       //
)

typedef velodyne_pcl::PointXYZIRT RTPoint;
typedef pcl::PointCloud<RTPoint> RTPointCloud;

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_pcl::PointXYZT, (float, x, x)  //
                                  (float, y, y)                           //
                                  (float, z, z)                           //
                                  (double, timestamp, timestamp)          //
)

typedef velodyne_pcl::PointXYZT PosPoint;
typedef pcl::PointCloud<PosPoint> PosCloud;

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_pcl::PointXYZIRPYT, (float, x, x)  //
                                  (float, y, y)                               //
                                  (float, z, z)                               //
                                  (float, intensity, intensity)               //
                                  (float, roll, roll)                         //
                                  (float, pitch, pitch)                       //
                                  (float, yaw, yaw)                           //
                                  (double, time, time)                        //
)

typedef velodyne_pcl::PointXYZIRPYT PosePoint;
typedef pcl::PointCloud<PosePoint> PosePointCloud;

typedef pcl::PointXYZRGB ColorPoint;
typedef pcl::PointCloud<ColorPoint> ColorPointCloud;

#endif
