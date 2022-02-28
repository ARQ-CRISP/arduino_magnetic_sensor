/*
 * Copyright: (C) 2019 CRISP, Advanced Robotics at Queen Mary,
 *                Queen Mary University of London, London, UK
 * Author: Rodrigo Neves Zenha <r.neveszenha@qmul.ac.uk>
 * CopyPolicy: Released under the terms of the GNU GPL v3.0.
 *
 */
/**
 * \file grid_map_plotter.cpp
 *
 * \author Rodrigo Neves Zenha
 * \copyright  Released under the terms of the GNU GPL v3.0.
 */

#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include "arduino_magnetic_sensor/xServerMsg.h"


#include <cmath>

using namespace grid_map;
using namespace std;

ros::Publisher publisher;


void ploterCallback(const arduino_magnetic_sensor::xServerMsg &msg)
{

  // Create grid map.
  GridMap map({"elevation", "normal_x", "normal_y", "normal_z"});
  map.setFrameId("map");
  map.setGeometry(Length(60, 60), 30);
  // map.setGeometry(Length(90, 180), 30);

  // Add data to grid map.
  ros::Time time = ros::Time::now();


  for (GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    Position position;
    map.getPosition(*it, position);
    Index index;
    map.getIndex(position, index);
    // cout << "Position: (" << index(0) << ","<< index(1) << ") -> msg.points index: " << (index(1))+ (index(0)) * 6 << endl;

    float vector_lenght = sqrt(pow((float)msg.points[(index(1))+ (index(0)) * 2].point.x, 2) + pow((float)msg.points[(index(1)) + (index(0)) * 6].point.y, 2) + pow((float)msg.points[(index(1)) + (index(0)) * 6].point.z, 2));

    map.at("elevation", *it) = (float)msg.points[(index(1)) + (index(0)) * 2].point.z;
    map.at("normal_x", *it) = (float)msg.points[(index(1)) + (index(0)) * 2].point.y * 10 / vector_lenght;
    map.at("normal_y", *it) = (float)msg.points[(index(1)) + (index(0)) * 2].point.x * 10 / vector_lenght;
    map.at("normal_z", *it) = (float)msg.points[(index(1)) + (index(0)) * 2].point.z * 10 / vector_lenght;

    // ROS_INFO("Printing node %s at position x:%i, y:%i, with value %f", msg.points[(index(1)) * 4 + (index(0))].header.frame_id.c_str(), index(0), index(1), msg.points[(index(1)) * 4 + (index(0))].point.z);
  }

// Publish grid map.
map.setTimestamp(time.toNSec());
grid_map_msgs::GridMap message;
GridMapRosConverter::toMessage(map, message);
publisher.publish(message);

}

int main(int argc, char **argv)
{
  // Initialize node and publisher.
  ros::init(argc, argv, "grid_map_ploter_node");
  ros::NodeHandle nh("~");

  publisher = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);

  ros::Subscriber sub = nh.subscribe("/xServTopic_calibrated", 1000, ploterCallback);

  ros::spin();
  // Wait for next cycle.

  return 0;
}
