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
#include "arduino_magnetic_sensor/xServerMsg.h"


#include <cmath>
// #include <math.h>

// Hardcoded values retrieved from trial & error -> Might need to be changed
#define XNODEMAXREAD 6000
#define YNODEMAXREAD 6000
#define ZNODEMAXREAD 9000
#define CONTACT_THERESHOLD 2 // Might need to be changed

using namespace std;

ros::Publisher calibrated_data_publisher, event_raw_data_publisher, event_calibrated_data_publisher; // Publish latest normalized sensor values

float uskin_pad_readings[4][3]; // Data structure holding current sensor reading values
float uskin_pad_min_reads[4][3] = {{65000, 65000, 65000},
                                    {65000, 65000, 65000},
                                    {65000, 65000, 65000},
                                    {65000, 65000, 65000}}; // Data structure holding minimum sensor reading values
int min_reads = 0;
bool normalized_flag = false;

// Contact contactDetection
bool contactDetection(arduino_magnetic_sensor::xServerMsg sensor_data)
{
  float data_diff[4][3];
  float mean = 0;
  float sum = 0;

  // Computing standard deviation accross all taxels and 3D channels
  for (int i = 0; i < 4; i++)
  {

    data_diff[i][0] = (float)sensor_data.points[i].point.x; // - uskin_pad_min_reads[i][0];
    data_diff[i][1] = (float)sensor_data.points[i].point.y; // - uskin_pad_min_reads[i][1];
    data_diff[i][2] = (float)sensor_data.points[i].point.z; // - uskin_pad_min_reads[i][2];

    mean += data_diff[i][0] + data_diff[i][1] + data_diff[i][2];
  }

  mean = mean / 12;

  for (int i = 0; i < 4; i++)
  {

    sum += pow(data_diff[i][0] - mean, 2);
    sum += pow(data_diff[i][1] - mean, 2);
    sum += pow(data_diff[i][2] - mean, 2);

  }

  sum = sum / 12;

  // Return true if std deviation if
  if (sqrt(sum) > CONTACT_THERESHOLD)
    return true;

  return false;

}

// Normalize current sensor values and store it in values_normalized
void normalize(float sensor_data[][3], arduino_magnetic_sensor::xServerMsg *normalized_sensor_data)
{

  // cout << "values before normalize: x: "<< sensor_data[0][0] << "  y: " << sensor_data[0][1] << "  z: " << sensor_data[0][2] << endl;
  // cout << "uskin min reads: "<< uskin_pad_min_reads[0][0] << "  y: " << uskin_pad_min_reads[0][1] << "  z: " << uskin_pad_min_reads[0][2] << endl;

  for (int index; index < 4; index++ )
  {
    arduino_magnetic_sensor::xSensorData taxel_data;
    taxel_data.taxels = index;
    taxel_data.point.x = (((sensor_data[index][0] - uskin_pad_min_reads[index][0]) / (XNODEMAXREAD - uskin_pad_min_reads[index][0])) * 100);
    taxel_data.point.y = (((sensor_data[index][1] - uskin_pad_min_reads[index][1]) / (YNODEMAXREAD - uskin_pad_min_reads[index][1])) * 100);
    taxel_data.point.z = (((sensor_data[index][2] - uskin_pad_min_reads[index][2]) / (ZNODEMAXREAD - uskin_pad_min_reads[index][2])) * 100);


    // Force normalized valies to be within boundaries 0 - 100
    taxel_data.point.z < 0 ? taxel_data.point.z = 0 : (taxel_data.point.z > 100 ? taxel_data.point.z = 100 : taxel_data.point.z);
    taxel_data.point.x < -100 ? taxel_data.point.x = -100 : (taxel_data.point.x > 100 ? taxel_data.point.x = 100 : taxel_data.point.x);
    taxel_data.point.y < -100 ? taxel_data.point.y = -100 : (taxel_data.point.y > 100 ? taxel_data.point.y = 100 : taxel_data.point.y);



    normalized_sensor_data->points.push_back(taxel_data);
  }

  // cout << "values before: x: "<< normalized_sensor_data->points[0].point.x << "  y: " << normalized_sensor_data->points[0].point.y << "  z: " << normalized_sensor_data->points[0].point.z << endl;

  return;
}

// ROS Subscribber callback to incoming sensor readings
void ploterCallback(const arduino_magnetic_sensor::xServerMsg &msg)
{

  // It is necessary to invert the signal
  for (int i = 0; i < 4; i++)
  {
    uskin_pad_readings[i][0] = -1 * (float)msg.points[i].point.x;
    uskin_pad_readings[i][1] = -1 * (float)msg.points[i].point.y;
    uskin_pad_readings[i][2] = -1 * (float)msg.points[i].point.z;
  }

  // Read the first 10 sensor values to determine minimum reading values
  // (or until data has been obtained from all 4 taxels)
  if (min_reads < 10 || !normalized_flag)
  {
    normalized_flag = true;

    // Store minimum reading values of all for taxels
    for (int i = 0; i < 4; i++)
    {
        // cout <<"Got values for taxel " << i << " are x: "<<  msg.points[i].point.x << "  y: " <<  msg.points[i].point.y << "  z: " <<  msg.points[i].point.z << endl;
        // msg.points[i].point.x != 0 is ignored since it represents sensor data loss
        if (uskin_pad_readings[i][0] < uskin_pad_min_reads[i][0] && uskin_pad_readings[i][0] != 0)
          uskin_pad_min_reads[i][0] = uskin_pad_readings[i][0];
        if (uskin_pad_readings[i][1] < uskin_pad_min_reads[i][1] && uskin_pad_readings[i][1] != 0)
          uskin_pad_min_reads[i][1] = uskin_pad_readings[i][1];
        if (uskin_pad_readings[i][2] < uskin_pad_min_reads[i][2] && uskin_pad_readings[i][2] != 0)
          uskin_pad_min_reads[i][2] = uskin_pad_readings[i][2];

        // If any of the taxels minimum readings has not yet been updated, more readings are necessary
        if(uskin_pad_min_reads[i][0] == 65000 || uskin_pad_min_reads[i][1] == 65000 || uskin_pad_min_reads[i][2] == 65000)
          normalized_flag = false;

        // cout <<"Min values for taxel " << i << " are x: "<< uskin_pad_min_reads[i][0] << "  y: " << uskin_pad_min_reads[i][1] << "  z: " << uskin_pad_min_reads[i][2] << endl;

    }
        cout << endl;
    min_reads++;
    // Current minimum readings obtained
    cout <<"Min values readings: " << min_reads << endl;

    return;
  }


  // Data structure holding latest normalized sensor values
  arduino_magnetic_sensor::xServerMsg uskin_pad;

  // Normalize sensor data for each taxel
  normalize(uskin_pad_readings, &uskin_pad);
  // Publish calibrated sensor data
  calibrated_data_publisher.publish(uskin_pad);

  if (contactDetection(uskin_pad))
  {
    // cout << "Contact has been detected!!" << endl;
    event_raw_data_publisher.publish(msg);
    event_calibrated_data_publisher.publish(uskin_pad);

  }


}

int main(int argc, char **argv)
{
  // Initialize node and publisher.
  ros::init(argc, argv, "calibrate_data");
  ros::NodeHandle nh("~");

  calibrated_data_publisher = nh.advertise<arduino_magnetic_sensor::xServerMsg>("/xServTopic_calibrated", 1, true);
  event_raw_data_publisher = nh.advertise<arduino_magnetic_sensor::xServerMsg>("/xServTopic_event", 1, true);
  event_calibrated_data_publisher = nh.advertise<arduino_magnetic_sensor::xServerMsg>("/xServTopic_calibrated_event", 1, true);

  ros::Subscriber sub = nh.subscribe("/xServTopic", 1000, ploterCallback);

  ros::spin();
  // Wait for next cycle.

  return 0;
}
