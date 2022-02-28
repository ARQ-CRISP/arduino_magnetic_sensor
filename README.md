# arduino_magnetic_sensor
ROS package - Magnetic Sensor Server for tactile data acquisition (using rosserial)

## List of dependencies (tested for ROS Kinetic distro)

### Rosserial - Launching ROS-compatible server on arduino for sensor data acquisition (http://wiki.ros.org/rosserial)
```console
sudo apt-get install ros-kinetic-rosserial -y
sudo apt-get install ros-kinetic-rosserial-arduino -y
```

### Grid Map - Used for data visualization purposes (http://wiki.ros.org/grid_map)     
```console
sudo apt-get install ros-kinetic-grid-map
```

#### Optionally, RQT_Plot can also be used for sensor data visualization (http://wiki.ros.org/rqt_plot)
```console
sudo apt-get install ros-kinetic-rqt
sudo apt-get install ros-kinetic-rqt-common-plugins
```
    
### Compiling the code
```console
catkin_make
catkin_make arduino_magnetic_sensor_firmware_server-upload
```
**NOTE**: The second command uploads the server code to the arduino board

