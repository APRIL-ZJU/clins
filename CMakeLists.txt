cmake_minimum_required(VERSION 2.8.3)
project(clins)

#SET(CXX_MARCH native)
#set(CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0  -march=${CXX_MARCH} -O3 -DBASALT_USE_CHOLMOD -DBASALT_DISABLE_ASSERTS -DNDEBUG")
#set(CMAKE_CXX_FLAGS " -march=${CXX_MARCH} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")
#add_definitions(-O3 -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    message_generation
    pcl_conversions
    pcl_ros
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    eigen_conversions
    tf_conversions
    roslib
)

find_package(Eigen3)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED )

find_package(yaml-cpp REQUIRED)
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  feature_cloud.msg
  imu_array.msg
  pose_array.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS std_msgs sensor_msgs message_runtime
#  LIBRARIES spline_test
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${YAML_CPP_LIBRARIES}
        ceres
)

################# Node ###################

add_executable(feature_extraction_node
               src/feature/feature_extraction_node.cpp
               src/feature/feature_extraction.cpp)
target_link_libraries(feature_extraction_node  ${thirdparty_libraries})
add_dependencies(feature_extraction_node  ${PROJECT_NAME}_generate_messages_cpp)

add_executable(clins_offline_node src/node/clins_offline_node.cpp
                                  src/odometry/inertial_initializer.cpp
                                  src/odometry/imu_state_estimator.cpp
                                  src/feature/feature_extraction.cpp)
target_link_libraries(clins_offline_node ${thirdparty_libraries})
add_dependencies(clins_offline_node  ${PROJECT_NAME}_generate_messages_cpp)

add_executable(kaist_sick_map_generation
               src/node/kaist_sick_map_generation.cpp)
target_link_libraries(kaist_sick_map_generation  ${thirdparty_libraries})

add_executable(recovery_vlp16_timestamp src/node/recovery_vlp16_timestamp.cpp)
target_link_libraries(recovery_vlp16_timestamp ${catkin_LIBRARIES} ${PCL_LIBRARIES})
