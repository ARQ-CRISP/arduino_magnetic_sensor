#include <Arduino.h>

#include <ros.h>
#include <ros/time.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>
#include "xSensorData.h"
#include "xServerMsg.h"

#include <Wire.h>


/**********************************************************************  VARIABLES DEFINITION ***********************************************************************************/
/********************************************************************************************************************************************************************************/

//Defining addresses of sensors
#define ADD0 0b00
#define ADD1 0b10 // Sensor 1 and 2 have their addresses swapped because of a connection error
#define ADD2 0b01
#define ADD3 0b11
#define sensor_base 0b1100
#define current_sensor ADD3

//Defining config modes (every sensor will have same config
#define BIST 0 // Built in self test, activates coil to calibrate and test. Reccomended 0
#define Z_SERIES 0 // All hall effect plates are only measuring Z axis (usful only when you don't care about X,Y axes)
#define GAIN_SEL 5 // A number between 0 and 7. The lower the number the less sensitive (different for XY and Z). Also dependent on RES bit, which is customizable for each axis.
#define HALLCONF 0xC // Hall plate spin adjustement. Unkown of its effects. Reccomended to leave at 0xC forever, and adjust other parameters instead.
#define TRIG_INT 1 // TRIGGER pin act as interruption or trigger (leaving at 0 in this use)
#define COMM 3 // 0,1 both modes. 2, only SPI. 3, only I2C. Will configure as I2C only
#define WOC_DIFF 0 // For wake up on change mode. 1 means it compares to previous value, 0 means it compares to initial value. Leave at 0 although mode probably will not be used
#define EXT_TRIG 0 // If this is 1 and TRIG_INT is 0, measurements will be made when trigger pin is enabled. Not in this case, since pin is NC
#define TCOMP 0 // Temperature compensation automatically made by chip. Sound nice but unfortunately transforms output in unsigned values. Not interested
#define BURST_SEL 0b1110 // Which axes are measuring in burst (zyxt), we'll not measure t. This register is useless since we'll use burst mode with zyxt in the command anyway
#define BURST_DATA_RATE 1 // 5 Bits that will define measurement rate as BURST_DATA_RATE*20ms. I believe a value of 0 means "as fast as possible?"
#define OSR2 0 // OSR is oversampling, means conversion will take more time (2^OSR2) but will have less noise. OSR2 is just for temperature we won't use it.
#define RESXYZ 0b010101 // In that order, sets the RES value for x,y,z. The higher, the more sensitive but values of 2 and 3 set output for unsigned int. We leave at 1. Can configure each axis independently
#define DIG_FILT 5 // A number between 0 and 7. Higher means slower but less noisy. Very low values are not allowed unless you change HALLCONF. We leave at 5 probably
#define OSR 0 // Read the part in OSR2 for explanation. This one regards the OSR in magnetic sensing. Leave at 0 unless too noisy.
// Additional register for sensitivity drift on different temperature than nominal (both over and under). This only matters if we're measuring magnetic valuesinstead of relative values.
//6 additional registers exist to configure offset and treshold for the WOC mode. None will be used for this application

/********************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************/

ros::NodeHandle node_handle;

arduino_magnetic_sensor::xServerMsg sensor_measurment_msg;
// arduino_magnetic_sensor::xSensorData single_taxel_measurment_msg[4];
// int16_t measurement[4][3];

ros::Publisher magnetic_sensor_publisher("xServTopic", &sensor_measurment_msg);

const byte sensors_array[] = {ADD0, ADD1, ADD3, ADD2};

// geometry_msgs::PoseArray array;
// ros::Publisher p("array", &array);

// int32_t message_counter_seq = 0;

void WireFlush()
{

  while(Wire.available()>0)
  {
    byte b = Wire.read(); // Flush I2C readings
  }

}

void Measure(byte address, int16_t* measurements)
{
  uint8_t data[7];
  // Request Measurement
  Wire.beginTransmission(address); // Transmit to address
  Wire.write(byte(0x4E)); // Measure everything except t
  Wire.endTransmission(false);  // Stop transmitting, but allows for repeated start.
  Wire.requestFrom(address,7,true); // Request 7 bytes (status + 2 per measurement) and then stops transmission
  Wire.endTransmission(true);
  // Read 7 bytes of data
  // status, xMag msb, xMag lsb, yMag msb, yMag lsb, zMag msb, zMag lsb
  if(Wire.available() == 7);
  {
    data[0] = Wire.read();
    data[1] = Wire.read();
    data[2] = Wire.read();
    data[3] = Wire.read();
    data[4] = Wire.read();
    data[5] = Wire.read();
    data[6] = Wire.read();
  }
  // Convert the data
  int16_t xMag = (data[1]<<8) | data[2];
  measurements[0] = xMag;
  int16_t yMag = (data[3]<<8) | data[4];
  measurements[1] = yMag;
  int16_t zMag = (data[5]<<8) | data[6];
  measurements[2] = zMag;
}

void MeasureAndPublish(byte address, bool retrieveAllSensors = false)
{

    if (retrieveAllSensors)
    {
        arduino_magnetic_sensor::xSensorData single_taxel_measurment_msg[4];
        int16_t measurement[4][3];

        for (int taxel_index = 0; taxel_index < 4; taxel_index++)
        {
            single_taxel_measurment_msg[taxel_index].taxels = taxel_index+1; // Taxel list should start from 1 and be sequential

            Measure(address|sensors_array[taxel_index], measurement[taxel_index]);
            single_taxel_measurment_msg[taxel_index].point.x = (float)measurement[taxel_index][1]; //x signal
            single_taxel_measurment_msg[taxel_index].point.y = (float)measurement[taxel_index][0]; //y signal
            single_taxel_measurment_msg[taxel_index].point.z = (float)measurement[taxel_index][2]; //z signal

        }

        sensor_measurment_msg.points_length = 4;
        sensor_measurment_msg.points = single_taxel_measurment_msg;

        magnetic_sensor_publisher.publish( &sensor_measurment_msg );


    }else{

        arduino_magnetic_sensor::xSensorData single_taxel_measurment_msg[1];
        // arduino_magnetic_sensor::xSensorData single_taxel_measurment_msg[4];
        int16_t measurement[3];

        // Measure(address|current_sensor, measurement);

        // for (int taxel_index = 0; taxel_index < 4; taxel_index++)
        // {
        //     single_taxel_measurment_msg[taxel_index].taxels = 1; // Taxel list should start from 1 and be sequential

        //     single_taxel_measurment_msg[taxel_index].point.x = (float)measurement[0]; //x signal
        //     single_taxel_measurment_msg[taxel_index].point.y = (float)measurement[1]; //y signal
        //     single_taxel_measurment_msg[taxel_index].point.z = (float)measurement[2]; //z signal

        // }


        // sensor_measurment_msg.points_length = 4;
        // sensor_measurment_msg.points = single_taxel_measurment_msg;

        // magnetic_sensor_publisher.publish( &sensor_measurment_msg );


        single_taxel_measurment_msg[0].taxels = 1;
        Measure(address|current_sensor, measurement);
        single_taxel_measurment_msg[0].point.x = (float)measurement[0]; //x signal
        single_taxel_measurment_msg[0].point.y = (float)measurement[1]; //y signal
        single_taxel_measurment_msg[0].point.z = (float)measurement[2]; //z signal

        sensor_measurment_msg.points_length = 1;
        sensor_measurment_msg.points= single_taxel_measurment_msg;
        magnetic_sensor_publisher.publish( &sensor_measurment_msg );

    }

}

void BurstIndividual(byte address)
{
  Wire.beginTransmission(address); // Transmit to address
  Wire.write(0b00011110); // Set to burst, zyx axes (no t)
  Wire.endTransmission(false);  // stop transmitting, but allows for repeated start.
  Wire.requestFrom(address,1,true); // Request 1 Byte (status byte) and then stops transmission
  Wire.endTransmission(true);
  if(Wire.available()>0)
    {
      byte statusbyte = Wire.read();
    }
}

void StartBurst()
{
  // Set sensors to Burst
  BurstIndividual(sensor_base|ADD0);
  BurstIndividual(sensor_base|ADD1);
  BurstIndividual(sensor_base|ADD2);
  BurstIndividual(sensor_base|ADD3);
  delay(100); // Wait a bit...
}

void WriteRegister(byte address, byte BH, byte BL, byte A) // Write register of sensor address, BH,BL and A is memory location
{

  Wire.beginTransmission(address);
  Wire.write(0x60); //...Write register command
  Wire.write(BH); // Write BH
  Wire.write(BL); // Write BL
  Wire.write(A<<2); // Where (shifted by 2)
  Wire.endTransmission(false); // Repeat Start
  Wire.requestFrom(address,1,true); // Request 1 Byte (status byte) and then stops transmission
  Wire.endTransmission(true);

  if(Wire.available()>0)
  {
    byte b = Wire.read();
    WireFlush();
  }

}

void WriteAllRegisters(byte BH, byte BL, byte A)
{

  WriteRegister(sensor_base|ADD0,BH,BL,A);
  WriteRegister(sensor_base|ADD1,BH,BL,A);
  WriteRegister(sensor_base|ADD2,BH,BL,A);
  WriteRegister(sensor_base|ADD3,BH,BL,A);

}

void ExitAll()
{

  Wire.beginTransmission(sensor_base|ADD0); // Transmit to address
  Wire.write(byte(0x80)); // Exit mode
  Wire.endTransmission(true);
  Wire.beginTransmission(sensor_base|ADD1); // Transmit to address
  Wire.write(byte(0x80)); // Exit mode
  Wire.endTransmission(true);
  Wire.beginTransmission(sensor_base|ADD2); // Transmit to address
  Wire.write(byte(0x80)); // Exit mode
  Wire.endTransmission(true);
  Wire.beginTransmission(sensor_base|ADD3); // Transmit to address
  Wire.write(byte(0x80)); // Exit mode
  Wire.endTransmission(true);
  delay(250);

}

void ConfigureSensors()
{

  // Configure register 0
  byte BH = 0x00 | BIST;
  byte BL = 0x00 | (Z_SERIES<<7);
  BL = BL | (GAIN_SEL<<4);
  BL = BL | HALLCONF;
  WriteAllRegisters(BH,BL,0);
  // Configure register 1
  BH = 0x00 | (TRIG_INT<<7);
  BH = BH | (COMM<<5);
  BH = BH | (WOC_DIFF<<4);
  BH = BH | (EXT_TRIG<<3);
  BH = BH | (TCOMP<<2);
  BH = BH | (BURST_SEL>>2);
  BL = 0x00 | (BURST_SEL<<6);
  BL = BL | BURST_DATA_RATE;
  WriteAllRegisters(BH,BL,1);
  //Configure register 2
  BH = 0x00 | (OSR2<<3);
  BH = BH | (RESXYZ>>3);
  BL = 0x00 | (RESXYZ<<5);
  BL = BL | (DIG_FILT<<2);
  BL = BL | OSR;
  WriteAllRegisters(BH,BL,2);

  ExitAll();
  WireFlush();
  delay(250);

}


void ResetAll()
{

  Wire.beginTransmission(sensor_base|ADD0); // Transmit to address
  Wire.write(byte(0xF0)); // Reset mode
  Wire.endTransmission(true);
  Wire.beginTransmission(sensor_base|ADD1); // Transmit to address
  Wire.write(byte(0xF0)); // Reset mode
  Wire.endTransmission(true);
  Wire.beginTransmission(sensor_base|ADD2); // Transmit to address
  Wire.write(byte(0xF0)); // Reset mode
  Wire.endTransmission(true);
  Wire.beginTransmission(sensor_base|ADD3); // Transmit to address
  Wire.write(byte(0xF0)); // Reset mode
  Wire.endTransmission(true);
  delay(250);

}



void setup()
{

  node_handle.initNode();
  node_handle.advertise(magnetic_sensor_publisher);
//   node_handle.advertise(p);


  Wire.begin(); // join i2c bus (address optional for master)
  // Deactivate internal pullups for wire (3v3).
  digitalWrite(SDA, 0);
  digitalWrite(SCL, 0);
// Serial.begin(9600); // start serial communication at 9600bps

  ExitAll();
  ResetAll();
  ConfigureSensors(); // COnfigures all sensors parameters
  StartBurst();
  delay(100);
}

void loop()
{
  //Output data to serial monitor
  // MeasureAndPublish(sensor_base);
  MeasureAndPublish(sensor_base, true);

  node_handle.spinOnce();
  delay(1);  // 1 measurement per sensor per second (aprox).
}
