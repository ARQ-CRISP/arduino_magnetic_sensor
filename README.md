# arduino_magnetic_sensor
ROS package - Magnetic Sensor Server for tactile data acquisition (using rosserial)

## List of dependencies (tested for ROS Kinetic distro)

### Rosserial - Launching ROS-compatible server on arduino for sensor data acquisition (http://wiki.ros.org/rosserial)
```console
sudo apt-get install ros-kinetic-rosserial -y
sudo apt-get install ros-kinetic-rosserial-arduino -y
```

#### Optionally, RQT_Plot can be used for sensor data visualization (http://wiki.ros.org/rqt_plot)
```console
sudo apt-get install ros-kinetic-rqt
sudo apt-get install ros-kinetic-rqt-common-plugins
```
    
### Compiling the code
```console
catkin_make
catkin_make arduino_magnetic_sensor_firmware_server-upload
```
**NOTE**: The second command uploads the server code to the arduino board. You do not need to run it if you simply wish to compile the package.

**NOTE**: If you run into issues uplaoding the arduino firmware: Be sure to have the arduino connected to the computer;  You might need to change the tty port to which the arduino is connected (e.g. USB0, ACM0) in the following files:
- firmware/CMakeLists.txt
- launch/start_server.launch
You can check the tty port to which the arduino is connected running the following command:
```console
ls -l /dev | grep tty
```

### Launching the code
```console
roslaunch arduino_magnetic_sensor start_server.launch
```
This file is responsible for launching the aruino (ros server) firmware, calibration ROS service node and visualiazation node (rqt plot).
Moreover, an optional node can be launched to record the ros messages being published to different topics. This is triggered by setting the "save_data" launch argument to 'true'. Aditional information can be given to name the resulting rosbag file. 

**NOTE**: This file can be edited to include/remove visualization and/or calibration nodes.


### Launching the visualization window
```console
roslaunch arduino_magnetic_sensor data_visualization.launch
```
If you are interested in simply launching the visualization tool, e.g. to check the contents of a previously record rosbag, the above command can be used.
To play a previosuly recorded rosbag simply run

```console
rosbag play <name_of_the_file.bag>
```

### Available Topics
After launching the server, the following topics are available to you:
- **/xServTopic**: continous raw sensor data;
- **/xServTopic_calibrated**: continuous calibrated sensor data; MinMax normalization;
- **/xServTopic_event**: Raw sensor data, published upon contact detection;
- **/xServTopic_calibrated_event**: calibrated sensor data, published upon contact detection;
- **/contact_decision**: continously publishing a single integer (1 or 0) representing whether contact is being detected (1 means contact);
- **/contact_thereshold**: continously publishing a single float representing the amount of data dispersion (compared to sensor at rest).

Only **/xServTopic** is published directly by the arduino. The remaining are provided by 'calibrate_data' node.



