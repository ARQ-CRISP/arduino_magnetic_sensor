cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)

include_directories(${ROS_LIB_DIR})

# Remove this if using an Arduino without native USB (eg, other than Leonardo)
add_definitions(-DUSB_CON)

# generate_arduino_firmware(test
#   SRCS test.cpp ${ROS_LIB_DIR}/time.cpp
#   BOARD uno
#   PORT /dev/ttyACM0
# )

generate_arduino_firmware(server
  SRCS arduino-server.cpp ${ROS_LIB_DIR}/time.cpp
  BOARD uno
  PORT /dev/ttyUSB0
  # PORT /dev/ttyACM0
)