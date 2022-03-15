cmake_minimum_required(VERSION 3.0.2)
project(arduino_magnetic_sensor)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  rosserial_arduino
  rosserial_client
  message_generation
  std_msgs
)

add_message_files(
  FILES
  xSensorData.msg
  xServerMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  roscpp
  std_msgs
  geometry_msgs
  message_runtime
)

rosserial_generate_ros_lib(
  PACKAGE rosserial_arduino
  SCRIPT make_libraries.py
)

# add_dependencies(server xela_server_generate_messages_cpp)


rosserial_configure_client(
  DIRECTORY firmware
  TOOLCHAIN_FILE ${ROSSERIAL_ARDUINO_TOOLCHAIN}
)

# rosserial_add_client_target(firmware test ALL)
# rosserial_add_client_target(firmware test-upload)

rosserial_add_client_target(firmware server ALL)
rosserial_add_client_target(firmware server-upload)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(
  calibrate_data
  src/calibrate_data.cpp
)

target_link_libraries(
  calibrate_data
  ${catkin_LIBRARIES}
)

## This makes sure message headers of this package are generated before being used.
add_dependencies(calibrate_data arduino_magnetic_sensor_generate_messages_cpp)

#############
## Install ##
#############

install(PROGRAMS
  scripts/temporal_viz.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
