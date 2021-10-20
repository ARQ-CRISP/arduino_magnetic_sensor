#ifndef _ROS_arduino_magnetic_sensor_xSensorData_h
#define _ROS_arduino_magnetic_sensor_xSensorData_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "geometry_msgs/Point.h"

namespace arduino_magnetic_sensor
{

  class xSensorData : public ros::Msg
  {
    public:
      typedef uint8_t _taxels_type;
      _taxels_type taxels;
      typedef geometry_msgs::Point _point_type;
      _point_type point;

    xSensorData():
      taxels(0),
      point()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->taxels >> (8 * 0)) & 0xFF;
      offset += sizeof(this->taxels);
      offset += this->point.serialize(outbuffer + offset);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      this->taxels =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->taxels);
      offset += this->point.deserialize(inbuffer + offset);
     return offset;
    }

    const char * getType(){ return "arduino_magnetic_sensor/xSensorData"; };
    const char * getMD5(){ return "1521ecb5350eb4c5ded8641d27de7b4c"; };

  };

}
#endif