import pandas as pd
import numpy as np
from collections import OrderedDict
import json




##################################################################
############################# Utils ##############################
##################################################################

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

##################################################################
################### Insert raw data/ labels ######################
##################################################################


def df_insertXaxisReadings(df, data, taxel_dict, taxels_list = None):

    if taxels_list is None:
        for taxel in taxel_dict['id_x_channel'].keys():
            # print(taxel)
            df[taxel]=data[:, taxel_dict['id_x_channel'][taxel], 0]

def df_insertYaxisReadings(df, data, taxel_dict, taxels_list = None):

    if taxels_list is None:
        for taxel in taxel_dict['id_y_channel'].keys():
            # print(taxel)
            df[taxel]=data[:, taxel_dict['id_y_channel'][taxel], 1]

def df_insertZaxisReadings(df, data, taxel_dict, taxels_list = None):

    if taxels_list is None:
        for taxel in taxel_dict['id_z_channel'].keys():
            # print(taxel)
            df[taxel]=data[:, taxel_dict['id_z_channel'][taxel], 2]


def df_insertXaxisReadings_calibrated(df, data, taxel_dict, taxels_list = None):

    if taxels_list is None:
        for taxel in taxel_dict['id_x_channel_calibrated'].keys():
            # print(taxel)
            df[taxel]=data[:, taxel_dict['id_x_channel_calibrated'][taxel], 0]

def df_insertYaxisReadings_calibrated(df, data, taxel_dict, taxels_list = None):

    if taxels_list is None:
        for taxel in taxel_dict['id_y_channel_calibrated'].keys():
            # print(taxel)
            df[taxel]=data[:, taxel_dict['id_y_channel_calibrated'][taxel], 1]

def df_insertZaxisReadings_calibrated(df, data, taxel_dict, taxels_list = None):

    if taxels_list is None:
        for taxel in taxel_dict['id_z_channel_calibrated'].keys():
            # print(taxel)
            df[taxel]=data[:, taxel_dict['id_z_channel_calibrated'][taxel], 2]



def df_insertLabels(df, data_dispersion, timestamps, age_labels, object_labels, experiments_label):

    df['Object']= object_labels
    df['Age']= age_labels
    df['Dispersion']= data_dispersion
    df['Experiment']= experiments_label
    df['Timestamps']= timestamps
    
    return


##################################################################
################### Build DataFrame #############################
##################################################################

def df_simple_build(df, data, data_calibrated, data_dispersion, timestamps, age_labels, object_labels, experiments_labels, signals = 'all'):
    print('>> df_simple_build')

    with open('../config/uskin4_tags.json') as json_file:
        can_tags = json.load(json_file, object_pairs_hook=OrderedDict)

    print('Signals is {}'.format(signals))
    
    if signals == 'normal_force':
        df_insertZaxisReadings(df, data, can_tags)
        df_insertZaxisReadings_calibrated(df, data_calibrated, can_tags)
    elif signals == 'shear_force':
        df_insertXaxisReadings(df,data, can_tags)
        df_insertYaxisReadings(df, data, can_tags)
        df_insertXaxisReadings_calibrated(df,data_calibrated, can_tags)
        df_insertYaxisReadings_calibrated(df, data_calibrated, can_tags)

    else:
        df_insertXaxisReadings(df, data, can_tags)
        df_insertYaxisReadings(df, data, can_tags)
        df_insertZaxisReadings(df, data, can_tags)
        df_insertXaxisReadings_calibrated(df, data_calibrated, can_tags)
        df_insertYaxisReadings_calibrated(df, data_calibrated, can_tags)
        df_insertZaxisReadings_calibrated(df, data_calibrated, can_tags)



    df_insertLabels(df, data_dispersion, timestamps, age_labels, object_labels, experiments_labels)

    print('<< df_simple_build')

##################################################################
############### Insert Aditional Features ########################
##################################################################

def df_insert_std(df, features_list, name='std'):
    print('>> df_insert_std with name: ' + name)
    
    if features_list is None or len(features_list) == 0:
        print('   Problems computing std features.')
        print('   It was not possible to read the list of features from which to compute the std')
        return
#     can_titles_x_diff+can_titles_y_diff+can_titles_z_diff
    df[name] = df.loc[:,features_list].std(axis=1)

    print('<< df_insert_std with name: ' + name)


def df_insert_diff(df, features_list, time_window=1, drop_feature=False):
    print('>> df_insert_diff with period: ' + str(time_window))
    
    if features_list is None or len(features_list) == 0:
        print('   Problems computing diff features.')
        print('   It was not possible to read the list of features from which to compute the diff')
        return

    for feature in features_list:
        df[feature+'_diff'] = (df.groupby(['Pose','Experiment', 'Slip', 'Object']))[feature].diff(periods=time_window).fillna(0.0)
        if drop_feature:
                df.drop(columns=[feature], inplace=True)
    print('<< df_insert_diff with period: ')