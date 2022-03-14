import numpy as np
import scipy.signal
import pandas as pd
import h5py


################################################
################### UTILS ######################
################################################

def print_exp(name): 
    
    if len(name.split('/')) == 4:
        print (name)
        
def print_attr(name): 
    if len(name.split('/')) > 4:
        if name.split('/')[1] == 'grape_1' and name.split('/')[3] == 'Exp1':
            print (name)
        return        


def open_files(path):
    
    hf_tactile = h5py.File(path, 'r')
    
    return hf_tactile


def close_files(hf_tactile):
    
    hf_tactile.close()

def tactile_label_statistics(labels):
    y = np.bincount(labels)
    ii = np.nonzero(y)[0]

    for a, b in zip(ii,y[ii]):
        print("Label {} appears {} times ({:.2f}%)".format(a, b, float(b)/float(labels.shape[0])*100))


###########################################################
########## Removing Grasp and Release poins  ##############
###########################################################

# def get_release_object_tactile_index(labels):
    
#     aux = np.argwhere(labels==2)
#     # For some experiments we don't have a release stage (unstable)
#     if aux.size > 0:
#         return aux.min()
#     else:
#         return labels.size

# def get_grasp_object_tactile_index(labels):

#     aux = np.argwhere(labels==3)

#     # If an error occurs here, something is wrong with the data labelling
#     if aux.size == 0:
#         print('Something seems wrong with the data tactile labelling!!')
#         print('No Grasp event detected!!')

#         return 0

#     return aux.max()+1

# # This can be used wither for raw, normalized tactile points or labels!
# def remove_grasp_release_points(x, tactile_labels, object_labels, timestamps, poses_labels, experiment_labels):

#     start_index = get_grasp_object_tactile_index(tactile_labels)
#     end_index = get_release_object_tactile_index(tactile_labels)
    
#     print('Trimming {} samples. Previous Indexes {}:{}. New Indexes {}:{}'.format(start_index+(tactile_labels.size-end_index),0, tactile_labels.size, start_index, end_index))
    
    
#     return x[:,:,start_index:end_index], tactile_labels[start_index:end_index], object_labels[start_index:end_index], timestamps[start_index:end_index], poses_labels[start_index:end_index], experiment_labels[start_index:end_index]



###########################################################
################### Manipulate Files ######################
###########################################################


def does_experiment_exist(hf_tactile, grasped_object, age_label, exp_num):
    
    try:
        if 'dragonskin_50_cilia/'+grasped_object+'/'+age_label+'/Exp'+str(exp_num) in hf_tactile:
            return True
        else:
            return False
    except: 
        print('Error: HDF5 file is not open!')
        return False


def get_data_raw(hf_tactile, grasped_object, age_label, exp_num):
    try:
        x = hf_tactile.get('dragonskin_50_cilia/'+grasped_object+'/'+age_label+'/Exp'+str(exp_num)+'/tactile_data')
    except: 
        print('Error: HDF5 file is not open!')
        return
        
    x = np.array(x)
    
    return x


def get_timestamps(hf_tactile, grasped_object, age_label, exp_num):
    try:
        timestamps = hf_tactile.get('dragonskin_50_cilia/'+grasped_object+'/'+age_label+'/Exp'+str(exp_num)+'/tactile_timestamps')
    except: 
        print('Error: HDF5 file is not open!')
        return
        
    timestamps = np.array(timestamps)
    
    return timestamps

def get_data_calibrated(hf_tactile, grasped_object, age_label, exp_num):
    try:
        x = hf_tactile.get('dragonskin_50_cilia/'+grasped_object+'/'+age_label+'/Exp'+str(exp_num)+'/tactile_data_calibrated')
    except: 
        print('Error: HDF5 file is not open!')
        return
        
    x = np.array(x)
    
    return x

def get_data_dispersion(hf_tactile, grasped_object, age_label, exp_num):
    try:
        x = hf_tactile.get('dragonskin_50_cilia/'+grasped_object+'/'+age_label+'/Exp'+str(exp_num)+'/tactile_data_dispersion')
    except: 
        print('Error: HDF5 file is not open!')
        return
        
    x = np.array(x)
    
    return x


### Get data, labels, and timestamps for 1 Object/ 1 Pose/ 1 Experiment
def get_experiment_data(hf_tactile, grasped_object, age_label, exp_num):

    if not does_experiment_exist(hf_tactile, grasped_object, age_label, exp_num):
        print('Experiment with key {} does not exist'.format(grasped_object+'/'+str(age_label)+'/Exp'+str(exp_num)))
        return None

    else: 
        x = get_data_raw(hf_tactile, grasped_object, age_label, exp_num)
        x_calibrated = get_data_calibrated(hf_tactile, grasped_object, age_label, exp_num)
        x_dispersion = get_data_dispersion(hf_tactile, grasped_object, age_label, exp_num)
        x_timestamps = get_timestamps(hf_tactile, grasped_object, age_label, exp_num)

        # tactile_labels = get_tactile_labels(hf_tactile,grasp_type, grasped_object,grasp_pose, exp_num)
        age_labels = np.repeat(age_label, np.size(x, axis=0)) # Array with object description\\\
        object_labels = np.repeat(grasped_object, np.size(x, axis=0)) # Array with object description\\\
        experiment_labels = np.repeat(exp_num, np.size(x, axis=0)) # Array with object description\\\
        
        # timestamps = get_timestamps(hf_tactile, grasp_type, grasped_object,grasp_pose, exp_num)
        
        print('Data has been loaded. ({} samples)'.format(np.size(x, axis=0)))

        return x, x_calibrated, x_dispersion, x_timestamps, age_labels, object_labels, experiment_labels


def get_experiment_tactile_labels_statistics(labels_list):
    
    print('\n==========================\n')
    print('Some labels statistics')
    for labels in labels_list:
        key,key_count = np.unique(labels, return_counts=True)
        # ii = np.nonzero(y)[0]

        # y = np.bincount(labels)
        # ii = np.nonzero(y)[0]


        for a, b in zip(key,key_count):
            print("Label "+str(a)+" appears {} times ({:.2f}%)".format(b, (b/labels.shape[0])*100))
        print('==========')
    print('\n==========================\n')

###########################################################
################### Load Data ######################
###########################################################
### Get data, labels, and timestamps for a list of:
# Objects;
# Poses;
# Experiment

def concatenate_data(original_array, new_array, axis):

    if original_array is np.empty:
        original_array = new_array
    else:
        original_array = np.append(original_array, np.array(new_array), axis=axis)

    return original_array

def get_complete_experimetal_data(data_path, object_list, exp_list=None):
    x = np.empty
    x_calibrated = np.empty
    x_dispersion = np.empty
    x_timestamps = np.empty
    age_labels = np.empty
    object_labels = np.empty
    experiment_labels = np.empty
    
    if exp_list is None or not exp_list:
        exp_list = range(0,20) # No especific experiments have been specified, so get them all
    
            
    tactile_file = open_files(data_path)
    
    for object_str in object_list: # possible grasp_poses
        for age_label in ['new', 'old']:
            print('Getting data for object: \'{}\', with age label: \'{}\''.format(object_str, age_label))
            for exp_num in exp_list: # possible exp_nums
            
                if does_experiment_exist(tactile_file, object_str, age_label, exp_num):

                    # x_temp, tactile_labels_temp, object_labels_temp, timestamps_temp, poses_labels_temp, experiments_labels_temp = get_experiment_data(tactile_file, grasp_type, object_str, grasp_pose, exp_num)
                    x_temp, x_calibrated_temp, x_dispersion_temp, x_timestamps_temp, age_labels_temp, object_labels_temp, experiment_labels_temp = get_experiment_data(tactile_file, 
                                                                                                                                                                        object_str, 
                                                                                                                                                                        age_label, 
                                                                                                                                                                        exp_num)
                    # # Check if there is a request for data to be truncated between grasp and release events
                    # if truncated:
                    #     x_temp, tactile_labels_temp, object_labels_temp, timestamps_temp, poses_labels_temp, experiments_labels_temp = remove_grasp_release_points(x_temp, tactile_labels_temp, object_labels_temp, timestamps_temp, poses_labels_temp, experiments_labels_temp)

                    x = concatenate_data(x, np.array(x_temp), axis=0)
                    x_calibrated = concatenate_data(x_calibrated, np.array(x_calibrated_temp), axis=0)
                    x_dispersion = concatenate_data(x_dispersion, np.array(x_dispersion_temp), axis=0)
                    x_timestamps = concatenate_data(x_timestamps, np.array(x_timestamps_temp), axis=0)
                    age_labels = concatenate_data(age_labels, np.array(age_labels_temp), axis = 0)
                    object_labels = concatenate_data(object_labels, np.array(object_labels_temp), axis = 0)
                    experiment_labels = concatenate_data(experiment_labels, np.array(experiment_labels_temp), axis = 0)
                else:
                    print('Failed to load data for experiment number {}'.format(exp_num))
                    break
    close_files(tactile_file)
    get_experiment_tactile_labels_statistics([age_labels, object_labels, experiment_labels])

    
    # return x_temp, slip_labels_temp, object_labels_temp, pose_labels_temp, experiment_labels_temp, sensor_labels_temp
    return x, x_calibrated, x_dispersion, x_timestamps, age_labels, object_labels, experiment_labels