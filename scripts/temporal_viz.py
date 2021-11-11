#!/usr/bin/env python
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py
import time

from numpy.core.records import record
from numpy.lib.function_base import diff

import rospy
from arduino_magnetic_sensor.msg import xServerMsg, xSensorData 
from std_msgs.msg import String


window_size = 160 # Defines number of consequetive samples to visualize/save 

# Use True if you wish visualization scale to bounded withing minimum/maximum readings for each experiment
# Otherwise scale will be fixed to boundaries described bellow
dynamicScale = False
X_MIN = -250
X_MAX = 250
Y_MIN = -100
Y_MAX = 100
Z_MIN = 0
Z_MAX = 800

x = np.arange(window_size)  # len = 10
y = np.arange(2)  # len = 6
X, Y = np.meshgrid(x, y)

last_sensor_readings = np.zeros([3, window_size])
sensor_readings_snapshot = np.zeros([3, window_size])
diff_sensor_readings = np.zeros([3, 1])
min_sensor_readings = np.zeros([3, 1])
max_sensor_readings = np.zeros([3, 1])
base_sensor_readings = np.zeros([3, 1])

is_first_reading = True
is_contact_detected = False

## Variables for data recording
record_data = False
recorded_sample = 0
save_data = False

## Publishers
pub_data_calibrated = rospy.Publisher("/xServTopic_calibrated", xServerMsg, queue_size=100)
pub_contact_detect = rospy.Publisher("/contactDetect", String, queue_size=10)
calibrated_data_taxel_msg = xSensorData()
calibrated_data_msg = xServerMsg()
calibrated_data_taxel_msg.taxels = 1
calibrated_data_msg.points.append(calibrated_data_taxel_msg)

## Experiment Details
object_size = ""
object_weight = ""
prototype = ""
meshplots_path = ""
hdf5_file_path = ""


# pick the desired colormap, sensible levels, and define a normalization 
# instance which takes data values and translates those into levels. 
cmap0 = plt.get_cmap('RdBu') 
cmap1 = plt.get_cmap('RdBu') 
cmap2 = plt.get_cmap('Greens')

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)


def groupExistsHDF5(experiment_key, hf):
    if experiment_key in hf:
        return True
    return False

def overwriteHDF5Data(experiment_key, hf):
    del hf[experiment_key]
    return

def saveH5pyFile(experiment_key, file_path, sequence, tactile_data):
    # Saving data
    hf = h5py.File(file_path+experiment_key+".h5", 'a')

    # Check if experiment has already been recorded in hdf5 file.
    if groupExistsHDF5(sequence, hf):
        # Confirm that user wants to overwrite data
        print("Warning, data will be overwritten")
        overwriteHDF5Data(sequence, hf)

    group_contact = hf.create_group(sequence)
    # group_vision = hf.create_group(experiment_key+'/Video')

    group_contact.create_dataset('grounded_data', data=np.array(tactile_data))

    hf.close()


def calibrate_sensor_reading(sensor_readings):
    global base_sensor_readings
    return sensor_readings - base_sensor_readings

def callback_read_tactile_data(msg):
    # print('>> callback_read_tactile_data')
    global last_sensor_readings
    global sensor_readings_snapshot
    global diff_sensor_readings
    global min_sensor_readings
    global max_sensor_readings
    global base_sensor_readings
    global is_first_reading
    global record_data
    global recorded_sample
    global save_data
    global is_contact_detected
    
    last_sensor_readings = np.roll(last_sensor_readings, 1, axis = 1)
    last_sensor_readings[0, 0] = msg.points[0].point.x
    last_sensor_readings[1, 0] = msg.points[0].point.y
    last_sensor_readings[2, 0] = msg.points[0].point.z

    last_sensor_readings[:,0] = -1*last_sensor_readings[:, 0]

    if is_first_reading: # 'Calibrate' following sensor readings by removing measurments with sensor at rest 
        base_sensor_readings = last_sensor_readings[:,0].copy() # Should we consider more readings?
        last_sensor_readings[:,0] = calibrate_sensor_reading(last_sensor_readings[:,0])
        min_sensor_readings = last_sensor_readings[:,0].copy()
        max_sensor_readings = last_sensor_readings[:,0].copy()
        print('min readings are {}'.format(min_sensor_readings))
        print('max readings are {}'.format(max_sensor_readings))
        is_first_reading = False
    else:
        # print('min readings {}'.format(min_sensor_readings))
        # print('max readings {}'.format(max_sensor_readings))
        # print('readings {}'.format(last_sensor_readings[:,-1]))

        last_sensor_readings[:,0] = calibrate_sensor_reading(last_sensor_readings[:,0])
        # print("last_sensor_readings: {}".format(last_sensor_readings[:,0]))
        calibrated_data_msg.points[0].point.x =  last_sensor_readings[0,0]
        calibrated_data_msg.points[0].point.y =  last_sensor_readings[1,0]
        calibrated_data_msg.points[0].point.z =  last_sensor_readings[2,0]
        pub_data_calibrated.publish(calibrated_data_msg)

        for i in range(3):
            if min_sensor_readings[i] > last_sensor_readings[i, 0]:
                if (i!=2 or last_sensor_readings[i, 0] > 0.0):
                    min_sensor_readings[i] = last_sensor_readings[i, 0].copy()
                    # print('new min reading {} for channel {}'.format(min_sensor_readings[i], i))
                    # print('min readings are {}'.format(min_sensor_readings))
                    # print('max readings are {}'.format(max_sensor_readings))
            elif max_sensor_readings[i] < last_sensor_readings[i, 0]:
                max_sensor_readings[i] = last_sensor_readings[i, 0].copy()
                # print('new max reading {} for channel {}'.format(max_sensor_readings[i], i))
                # print('min readings are {}'.format(min_sensor_readings))
                # print('max readings are {}'.format(max_sensor_readings))

    diff_sensor_readings = last_sensor_readings[:, 0] - last_sensor_readings[:, 1]
    # print (diff_sensor_readings)
    # print(diff_sensor_readings)
    # print(np.std(diff_sensor_readings))
    if np.std(diff_sensor_readings) > 7:
        # print('>>CONTACT DETECTED!')
        pub_contact_detect.publish(String("CONTACT DETECTED"))
        # print(diff_sensor_readings[2]) 
        if not record_data and diff_sensor_readings[2] > 0: # Assuming difference in Z channel will be negative when object is "lifted" from the sensor
            print(">> Pressing Contact detected - Will attempt to save data now")
            # is_contact_detected = True
            record_data = True
    # else:
        # is_contact_detected = False

    if record_data:
        recorded_sample = recorded_sample + 1
        if recorded_sample == window_size:
            # print('here')
            recorded_sample = 0
            record_data = False
            save_data = True
            sensor_readings_snapshot = last_sensor_readings.copy()
            # plt.savefig('/home/rodrigo/Documents/github/catkin_magnetic_sensor/src/arduino_magnetic_sensor/scripts/temp.png', dpi=fig.dpi)


    #     print('new min readings {}'.format(min_sensor_readings))
    #     print('new max readings {}'.format(max_sensor_readings))

    # print(last_sensor_readings)


def init_services():

    global object_size
    global object_weight
    global prototype
    global meshplots_path
    global hdf5_file_path
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.Subscriber("/xServTopic", xServerMsg, callback_read_tactile_data)

    if rospy.has_param('/temporal_viz/object_size'):
        object_size = rospy.get_param(
            "/temporal_viz/object_size")

        print("Sucessfully read object_size: {}".format(object_size))
    else:
        print('Unable to find object_size')

    if rospy.has_param('/temporal_viz/object_weight'):
        object_weight = rospy.get_param(
            "/temporal_viz/object_weight")

        print('Sucessfully read object_weight: {}'.format(object_weight))
    else:
        print('Unable to find object_weight')

    if rospy.has_param('/temporal_viz/prototype'):
        prototype = rospy.get_param(
            "/temporal_viz/prototype")

        print('Sucessfully read prototype {}'.format(prototype))
    else:
        print('Unable to find prototype')

    if rospy.has_param('/temporal_viz/meshplots_path'):
        meshplots_path = rospy.get_param(
            "/temporal_viz/meshplots_path")

        print('Sucessfully read meshplots_path {}'.format(meshplots_path))
    else:
        print('Unable to find meshplots_path')

    if rospy.has_param('/temporal_viz/hdf5_file_path'):
        hdf5_file_path = rospy.get_param(
            "/temporal_viz/hdf5_file_path")

        print('Sucessfully read hdf5_file_path {}'.format(hdf5_file_path))
    else:
        print('Unable to find hdf5_file_path')
        


if __name__ == '__main__':
    # global save_data 
    
    rospy.init_node('temporal_viz', anonymous=True)

    init_services()

    plt.ion()
    plt.show()

    fig.patch.set_facecolor('lightgrey')    
 

    # x and y are bounds, so z should be the value *inside* those bounds. 
    # Therefore, remove the last value from the z array.
    if dynamicScale: 
        levels0 = MaxNLocator(nbins=window_size/10).tick_values(last_sensor_readings[0].min(), last_sensor_readings[0].max()) 
        levels1= MaxNLocator(nbins=window_size/10).tick_values(last_sensor_readings[1].min(), last_sensor_readings[1].max()) 
        levels2 = MaxNLocator(nbins=window_size).tick_values(last_sensor_readings[2].min(), last_sensor_readings[2].max())
    else:
        levels0 = MaxNLocator(nbins=window_size/10).tick_values(X_MIN, X_MAX) 
        levels1= MaxNLocator(nbins=window_size/10).tick_values(Y_MIN, Y_MAX) 
        levels2 = MaxNLocator(nbins=window_size).tick_values(Z_MIN, Z_MAX)


    # print("levels0 >> {}".format(levels0))
    # print("levels1 >> {}".format(levels1))
    # print("levels2 >> {}".format(levels2))

    norm0 = BoundaryNorm(levels0, ncolors=cmap0.N, clip=True)
    norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
    norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

    
    im0 = ax0.pcolormesh(X, Y, np.tile(last_sensor_readings[0], (2,1)), cmap=cmap0, norm=norm0) # np.tile repeats readings row (pcolormesh requires matri data)
    cbar0 = fig.colorbar(im0, ax=ax0) 
    ax0.set_title('Channel X') 

    im1 = ax1.pcolormesh(X, Y, np.tile(last_sensor_readings[1], (2,1)), cmap=cmap1, norm=norm1) # np.tile repeats readings row (pcolormesh requires matri data)
    cbar1 = fig.colorbar(im1, ax=ax1) 
    ax1.set_title('Channel Y') 

    im2 = ax2.pcolormesh(X, Y, np.tile(last_sensor_readings[2], (2,1)), cmap=cmap2, norm=norm2) # np.tile repeats readings row (pcolormesh requires matri data)
    cbar2 = fig.colorbar(im2, ax=ax2) 
    ax2.set_title('Channel Z') 

    ax2.set_xlabel('Number of previous samples')
    ax0.axes.yaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    fig.suptitle('Pcolormesh of the sensor last '+str(window_size)+' samples', fontsize=16)


    plt.draw()
    plt.pause(0.001)

    figure_sequence = 1

    while not rospy.is_shutdown():
        # x and y are bounds, so z should be the value *inside* those bounds. 
        # Therefore, remove the last value from the z array.
    

        # print("min0: {}, max0 {}".format(min_sensor_readings[0], min_sensor_readings[0]))
        # print("min1: {}, max1 {}".format(min_sensor_readings[1], max_sensor_readings[1]))
        # print("min2: {}, max2 {}".format(min_sensor_readings[2], max_sensor_readings[2]))

        if dynamicScale:
            levels0 = MaxNLocator(nbins=window_size/10).tick_values(min_sensor_readings[0], max_sensor_readings[0]) 
            levels1= MaxNLocator(nbins=window_size/10).tick_values(min_sensor_readings[1],  max_sensor_readings[1]) 
            levels2 = MaxNLocator(nbins=window_size).tick_values(min_sensor_readings[2], max_sensor_readings[2])
        else:
            levels0 = MaxNLocator(nbins=window_size/10).tick_values(X_MIN, X_MAX) 
            levels1= MaxNLocator(nbins=window_size/10).tick_values(Y_MIN, Y_MAX) 
            levels2 = MaxNLocator(nbins=window_size).tick_values(Z_MIN, Z_MAX)

        norm0 = BoundaryNorm(levels0, ncolors=cmap0.N, clip=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        if save_data:
            print("  >> Saving figure for contact number {}".format(figure_sequence))
            im0 = ax0.pcolormesh(X, Y, np.tile(sensor_readings_snapshot[0], (2,1)), cmap=cmap0, norm=norm0)  # np.tile repeats readings row (pcolormesh requires matri data)
            cbar0.update_bruteforce(im0)
            im1 = ax1.pcolormesh(X, Y, np.tile(sensor_readings_snapshot[1], (2,1)), cmap=cmap1, norm=norm1) # np.tile repeats readings row (pcolormesh requires matri data)
            cbar1.update_bruteforce(im1)
            im2 = ax2.pcolormesh(X, Y, np.tile(sensor_readings_snapshot[2], (2,1)), cmap=cmap2, norm=norm2) # np.tile repeats readings row (pcolormesh requires matri data)
            cbar2.update_bruteforce(im2)

            exp_name_str = "Experiment_"+str(object_size)+"_"+str(object_weight)+"_"+str(prototype)
            plt.savefig(meshplots_path+exp_name_str+"_"+str(figure_sequence)+'.png', dpi=fig.dpi)
            save_data = False
            print("  << Figure has been saved")

            print("  >> Saving data in HDF5 file")
            saveH5pyFile(exp_name_str, hdf5_file_path, str(figure_sequence), sensor_readings_snapshot)
            print("  << Data has been saved")

            figure_sequence += 1

        else:
            im0 = ax0.pcolormesh(X, Y, np.tile(last_sensor_readings[0], (2,1)), cmap=cmap0, norm=norm0)  # np.tile repeats readings row (pcolormesh requires matri data)
            cbar0.update_bruteforce(im0)
            im1 = ax1.pcolormesh(X, Y, np.tile(last_sensor_readings[1], (2,1)), cmap=cmap1, norm=norm1) # np.tile repeats readings row (pcolormesh requires matri data)
            cbar1.update_bruteforce(im1)
            im2 = ax2.pcolormesh(X, Y, np.tile(last_sensor_readings[2], (2,1)), cmap=cmap2, norm=norm2) # np.tile repeats readings row (pcolormesh requires matri data)
            cbar2.update_bruteforce(im2)

        plt.draw()
        plt.pause(0.001)

    rospy.spin()

