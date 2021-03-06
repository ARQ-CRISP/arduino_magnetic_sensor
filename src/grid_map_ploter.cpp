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

// Hardcoded values retrieved from trial & error
#define XNODEMAXREAD 5000
#define YNODEMAXREAD 5000
#define ZNODEMAXREAD 5000

using namespace grid_map;
using namespace std;

ros::Publisher publisher;
int sensor_to_visualize;

float uskin_pad[4][3];
float uskin_pad_min_reads[4][3] = {{65000, 65000, 65000},
                                    {65000, 65000, 65000},
                                    {65000, 65000, 65000},
                                    {65000, 65000, 65000}};
int min_reads = 0;
bool normalized_flag = false;


void normalize(int x, int y, int z, int index, float values_normalized[24][3])
{
  // if (index == 0){
  //   cout << "values before normalize: x: "<< x << "  y: " << y << "  z: " << z << endl;
  //   cout << "uskin min reads: "<< uskin_pad_min_reads[0][0] << "  y: " << uskin_pad_min_reads[0][1] << "  z: " << uskin_pad_min_reads[0][2] << endl;
  // }

  values_normalized[index][0] = (int)((((float)x - uskin_pad_min_reads[index][0]) / (XNODEMAXREAD - uskin_pad_min_reads[index][0])) * 100);
  values_normalized[index][1] = (int)((((float)y - uskin_pad_min_reads[index][1]) / (YNODEMAXREAD - uskin_pad_min_reads[index][1])) * 100);
  values_normalized[index][2] = (int)((((float)z - uskin_pad_min_reads[index][2]) / (ZNODEMAXREAD - uskin_pad_min_reads[index][2])) * 100);
  // if (z_value < 0)
  //   z_value = 0; // Force Z to be above 0. Z would only get negative values if a node is being "pulled", which should not happen
  // Force normalized valies to be within boundaries
  values_normalized[index][2] < 0 ? values_normalized[index][2] = 0 : (values_normalized[index][2] > 100 ? values_normalized[index][2] = 100 : values_normalized[index][2]);
  values_normalized[index][0] < -100 ? values_normalized[index][0] = -100 : (values_normalized[index][0] > 100 ? values_normalized[index][0] = 100 : values_normalized[index][0]);
  values_normalized[index][1] < -100 ? values_normalized[index][1] = -100 : (values_normalized[index][1] > 100 ? values_normalized[index][1] = 100 : values_normalized[index][1]);

  // if (index == 0)
  // cout << "values before: x: "<< values_normalized[0][0] << "  y: " << values_normalized[0][1] << "  z: " << values_normalized[0][2] << endl;

  return;
}

void ploterCallback(const arduino_magnetic_sensor::xServerMsg &msg)
{

  // Create grid map.
  GridMap map({"elevation", "normal_x", "normal_y", "normal_z"});
  map.setFrameId("map");
  map.setGeometry(Length(60, 60), 30);
  // map.setGeometry(Length(90, 180), 30);

  // Add data to grid map.
  ros::Time time = ros::Time::now();

  // cout << min_reads<< endl;

  if (min_reads < 10 || !normalized_flag)
  {
    normalized_flag = true;
    for (int i = 0; i < 4; i++)
    {
        // cout <<"Got values for taxel " << i << " are x: "<<  msg.points[i].point.x << "  y: " <<  msg.points[i].point.y << "  z: " <<  msg.points[i].point.z << endl;
        if (msg.points[i].point.x < uskin_pad_min_reads[i][0] && msg.points[i].point.x != 0)
          uskin_pad_min_reads[i][0] = msg.points[i].point.x;
        if (msg.points[i].point.y < uskin_pad_min_reads[i][1] && msg.points[i].point.y != 0)
          uskin_pad_min_reads[i][1] = msg.points[i].point.y;
        if (msg.points[i].point.z < uskin_pad_min_reads[i][2] && msg.points[i].point.z != 0)
          uskin_pad_min_reads[i][2] = msg.points[i].point.z;

        if(uskin_pad_min_reads[i][0] == 65000 || uskin_pad_min_reads[i][1] == 65000 || uskin_pad_min_reads[i][2] == 65000)
          normalized_flag = false;

        // cout <<"Min values for taxel " << i << " are x: "<< uskin_pad_min_reads[i][0] << "  y: " << uskin_pad_min_reads[i][1] << "  z: " << uskin_pad_min_reads[i][2] << endl;

    }
        cout << endl;
    min_reads++;
    cout <<"Min values readings: " << min_reads << endl;

    return;
  }


  for (int i = 0; i < 4; i++)  
    normalize(msg.points[i].point.x, msg.points[i].point.y, msg.points[i].point.z, i, uskin_pad);
 



  for (GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    Position position;
    map.getPosition(*it, position);
    Index index;
    map.getIndex(position, index);
    // cout << "Position: (" << index(0) << ","<< index(1) << ") -> uskin_pad index: " << (index(1))+ (index(0)) * 6 << endl;

    float vector_lenght = sqrt(pow(uskin_pad[(index(1))+ (index(0)) * 2][0], 2) + pow(uskin_pad[(index(1)) + (index(0)) * 6][1], 2) + pow(uskin_pad[(index(1)) + (index(0)) * 6][2], 2));

    map.at("elevation", *it) = -1 * uskin_pad[(index(1)) + (index(0)) * 2][2];
    map.at("normal_x", *it) = -1 * uskin_pad[(index(1)) + (index(0)) * 2][1] * 10 / vector_lenght;
    map.at("normal_y", *it) = -1 * uskin_pad[(index(1)) + (index(0)) * 2][0] * 10 / vector_lenght;
    map.at("normal_z", *it) = uskin_pad[(index(1)) + (index(0)) * 2][2] * 10 / vector_lenght;

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

  ros::Subscriber sub = nh.subscribe("/xServTopic", 1000, ploterCallback);

  ros::spin();
  // Wait for next cycle.

  return 0;
}
