#!/usr/bin/env python
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import time

from numpy.core.records import record

import rospy
from arduino_magnetic_sensor.msg import xServerMsg


window_size = 160 # Defines number of consequetive samples to visualize/save 

x = np.arange(window_size)  # len = 10
y = np.arange(2)  # len = 6
X, Y = np.meshgrid(x, y)

last_sensor_readings = np.zeros([3, window_size])
diff_sensor_readings = np.zeros([3, 1])
min_sensor_readings = np.zeros([3, 1])
max_sensor_readings = np.zeros([3, 1])
base_sensor_readings = np.zeros([3, 1])
is_first_reading = True

## Variables for data recording
record_data = False
recorded_sample = 0

# pick the desired colormap, sensible levels, and define a normalization 
# instance which takes data values and translates those into levels. 
cmap0 = plt.get_cmap('RdBu') 
cmap1 = plt.get_cmap('PiYG') 
cmap2 = plt.get_cmap('Greens')

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3) 


def calibrate_sensor_reading(sensor_readings):
    global base_sensor_readings
    return sensor_readings - base_sensor_readings

def callback_read_tactile_data(msg):
    # print('>> callback_read_tactile_data')
    global last_sensor_readings
    global diff_sensor_readings
    global min_sensor_readings
    global max_sensor_readings
    global base_sensor_readings
    global is_first_reading
    global record_data
    global recorded_sample
    
    last_sensor_readings = np.roll(last_sensor_readings, 1, axis = 1)
    last_sensor_readings[0, 0] = msg.points[0].point.x
    last_sensor_readings[1, 0] = msg.points[0].point.y
    last_sensor_readings[2, 0] = msg.points[0].point.z

    last_sensor_readings[:,0] = -1*last_sensor_readings[:, 0]

    if is_first_reading:
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

        for i in range(3):
            if min_sensor_readings[i] > last_sensor_readings[i, 0]:
                if (i!=2 or last_sensor_readings[i, 0] > 0.0):
                    min_sensor_readings[i] = last_sensor_readings[i, 0].copy()
                    print('new min reading {} for channel {}'.format(min_sensor_readings[i], i))
            elif max_sensor_readings[i] < last_sensor_readings[i, 0]:
                max_sensor_readings[i] = last_sensor_readings[i, 0].copy()
                print('new max reading {} for channel {}'.format(max_sensor_readings[i], i))


    diff_sensor_readings = last_sensor_readings[:, 0] - last_sensor_readings[:, 1]
    # print (diff_sensor_readings)
    # print(diff_sensor_readings)
    # print(np.std(diff_sensor_readings))
    if np.std(diff_sensor_readings) > 7:
        print('CONTACT DETECTED!')
        # if not record_data: # Otherwise, data is already being recorded
        record_data = True

    # if record_data:
    #     recorded_sample = recorded_sample + 1
    #     # print(recorded_sample)
    #     if recorded_sample == window_size:
    #         # print('here')
    #         recorded_sample = 0
    #         record_data = False
    #         plt.savefig('/home/rodrigo/Documents/github/catkin_magnetic_sensor/src/arduino_magnetic_sensor/scripts/temp.png', dpi=fig.dpi)

        # print('new min readings {}'.format(min_sensor_readings))
        # print('new max readings {}'.format(max_sensor_readings))

    # print(last_sensor_readings)


def init_services():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.Subscriber("/xServTopic", xServerMsg, callback_read_tactile_data)


if __name__ == '__main__':
    
    rospy.init_node('temporal_viz', anonymous=True)

    init_services()

    plt.ion()
    plt.show()

    fig.patch.set_facecolor('lightgrey')    
 

    # x and y are bounds, so z should be the value *inside* those bounds. 
    # Therefore, remove the last value from the z array. 
    levels0 = MaxNLocator(nbins=window_size/10).tick_values(last_sensor_readings[0].min(), last_sensor_readings[0].max()) 
    levels1= MaxNLocator(nbins=window_size/10).tick_values(last_sensor_readings[1].min(), last_sensor_readings[1].max()) 
    levels2 = MaxNLocator(nbins=window_size).tick_values(last_sensor_readings[2].min(), last_sensor_readings[2].max()) 

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

    while not rospy.is_shutdown():
        # x and y are bounds, so z should be the value *inside* those bounds. 
        # Therefore, remove the last value from the z array.
    

        # print("min0: {}, max0 {}".format(min_sensor_readings[0], min_sensor_readings[0]))
        # print("min1: {}, max1 {}".format(min_sensor_readings[1], max_sensor_readings[1]))
        # print("min2: {}, max2 {}".format(min_sensor_readings[2], max_sensor_readings[2]))

        levels0 = MaxNLocator(nbins=window_size/10).tick_values(min_sensor_readings[0], max_sensor_readings[0]) 
        levels1= MaxNLocator(nbins=window_size/10).tick_values(min_sensor_readings[1],  max_sensor_readings[1]) 
        levels2 = MaxNLocator(nbins=window_size).tick_values(min_sensor_readings[2], max_sensor_readings[2])
        
        # levels0 = MaxNLocator(nbins=15).tick_values(last_sensor_readings[0].min(), last_sensor_readings[0].max()) 
        # levels1= MaxNLocator(nbins=15).tick_values(last_sensor_readings[1].min(), last_sensor_readings[1].max()) 
        # levels2 = MaxNLocator(nbins=15).tick_values(last_sensor_readings[2].min(), last_sensor_readings[2].max()) 

        # print("levels0 >> {}".format(levels0))
        # print("levels1 >> {}".format(levels1))
        # print("levels2 >> {}".format(levels2))

        norm0 = BoundaryNorm(levels0, ncolors=cmap0.N, clip=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        im0 = ax0.pcolormesh(X, Y, np.tile(last_sensor_readings[0], (2,1)), cmap=cmap0, norm=norm0)  # np.tile repeats readings row (pcolormesh requires matri data)
        cbar0.update_bruteforce(im0)
        im1 = ax1.pcolormesh(X, Y, np.tile(last_sensor_readings[1], (2,1)), cmap=cmap1, norm=norm1) # np.tile repeats readings row (pcolormesh requires matri data)
        cbar1.update_bruteforce(im1)
        im2 = ax2.pcolormesh(X, Y, np.tile(last_sensor_readings[2], (2,1)), cmap=cmap2, norm=norm2) # np.tile repeats readings row (pcolormesh requires matri data)
        cbar2.update_bruteforce(im2)


        # print('Ola')
        plt.draw()
        plt.pause(0.001)

    rospy.spin()

