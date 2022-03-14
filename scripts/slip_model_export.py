from matplotlib import use
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import load_exp
import pandas_df_manipulation as pdm


import seaborn as sns

from collections import OrderedDict
import json
from joblib import dump

import matplotlib.pyplot as plt

number_of_trials = 5
experiment_type = "val_and_test"
use_pca= True
max_principal_components = 0

ml_models = ["random_forest", "SVM", "AdaBoost", "MLP_default", "MLP_1", "MLP_2", "KNN"]

CTEs_test = [None] * len(ml_models)
error_test = [None] * len(ml_models)
CTEs_val = [None] * len(ml_models)
error_val = [None] * len(ml_models)

f1_scores_test = dict.fromkeys(ml_models)
f1_scores_val = dict.fromkeys(ml_models)

for key in f1_scores_test.keys():
	f1_scores_test[key] = np.zeros(number_of_trials)
	f1_scores_val[key] = np.zeros(number_of_trials)

def load_json_config_file(path):

    with open(path) as json_file:
        params = json.load(json_file, object_pairs_hook=OrderedDict)

    return params

def load_train_validation_data(data_config_file_path):
    
    can_tags = load_json_config_file('../config/uskin4_tags.json') # Should not need to change
    train_data_params = load_json_config_file(data_config_file_path) # Full path with file name

    data_path = train_data_params['data_path']
    # grasp_method = train_data_params['grasp_method']
    window_size = train_data_params['window_size'] #None or any value
    data_split = train_data_params['data_split'] #Only Validation or Validation and Test
    
    train_object_list = train_data_params['train_object_list']
    test_object_list = train_data_params['test_object_list']
    train_experiments = train_data_params['train_experiments']
    test_experiments = train_data_params['test_experiments']
    
    rf_max_depth = train_data_params['rf_max_depth']
    rf_decision_trees = train_data_params['rf_decision_trees']
    classification_prob_thereshold = train_data_params['classification_prob_thereshold']
    
    print('####### Experimental INFO (Record this) ########')
    print('Training and testing data split type: {}'.format(data_split))
    # print('Training data balance: {}'.format(data_balancing))
    

    df = pd.DataFrame()

    # We start by loading all objects, exp and poses required for the training and testing sets
    objects_to_load = train_object_list + list(set(test_object_list)-set(train_object_list))
    experiments_to_load = (train_experiments + list(set(test_experiments)-set(train_experiments))).sort()


    x, x_calibrated, x_dispersion, x_timestamps, age_labels, object_labels, experiment_labels = load_exp.get_complete_experimetal_data(data_path, 
                                                                                           objects_to_load,
                                                                                           exp_list=experiments_to_load)

    age_labels = np.array((age_labels == "new").astype(int), dtype=int)
    print(x.shape)
    load_exp.tactile_label_statistics(age_labels)

    pdm.df_simple_build(df, x, x_calibrated, x_dispersion, x_timestamps, age_labels, object_labels, experiment_labels, signals = 'normal_force')
    
    # diff_features_list = list(can_tags['id_x_channel'].keys())+list(can_tags['id_y_channel'].keys())+list(can_tags['id_z_channel'].keys())
    # std_features_list = can_tags['id_x_channel_diff']+can_tags['id_y_channel_diff']+can_tags['id_z_channel_diff'] 
    
    # pdm.df_insert_diff(df, features_list = diff_features_list, time_window=window_size)
    # pdm.df_insert_std(df, features_list = std_features_list)

    if data_split == 'val':

        X_train, X_val, Y_train, Y_val = train_test_split(df.loc[:,df.columns.difference(['Age'], sort=False)], df['Age'], test_size=0.33, stratify=df['Age'])
        # X_train, X_val, Y_train, Y_val = train_test_split(df.loc[:,df.columns.difference(['Slip'], sort=False)], df['Slip'], test_size=0.33, stratify=df['Slip'], random_state=123456)

        print('Train data distribution')
        load_exp.tactile_label_statistics(Y_train)
        load_exp.get_experiment_tactile_labels_statistics([X_train['Object'], X_train['Experiment']])
        print('Validation data distribution')
        load_exp.tactile_label_statistics(Y_val)
        load_exp.get_experiment_tactile_labels_statistics([X_val['Object'], X_val['Experiment']])


        if test_object_list is not None and len(test_object_list) != 0:
            print('Only Validating data from objects: {}'.format(test_object_list))
            object_select_index = X_val.loc[X_val['Object'].isin(test_object_list)].index
            X_val = X_val.loc[object_select_index,:]
            Y_val = Y_val.loc[object_select_index]
            load_exp.tactile_label_statistics(Y_val)
            

        # Remove 'Object', 'Experiment', 'Pose', and 'Sensor' columns
        X_train = X_train.loc[:,X_train.columns.difference(['Object', 'Experiment', 'Timestamps', '0_X', '1_X', '2_X', '3_X', '0_Y', '1_Y', '2_Y', '3_Y', '0_Z', '1_Z', '2_Z', '3_Z'])]
        X_val = X_val.loc[:,X_val.columns.difference(['Object', 'Experiment', 'Timestamps', '0_X', '1_X', '2_X', '3_X', '0_Y', '1_Y', '2_Y', '3_Y', '0_Z', '1_Z', '2_Z', '3_Z'])]

            
        print('Validating all remaning data')
        return X_train, X_val, [], Y_train, Y_val, [], rf_max_depth, rf_decision_trees, classification_prob_thereshold

    elif data_split == 'val_and_test':
        #### DF rows shuffling ####
        df = df.sample(frac=1)
        
        print("##########Building train dataset....... ")
        #### Extract X_train, Y_train ####
        #### Both 'Object' and 'Object Slip' tags are removed
        X_train = df.loc[(df['Object'].isin(list(set(train_object_list)-set(test_object_list)))) & (df['Experiment'].isin(train_experiments))]
        # X_train = df.loc[(df['Object'].isin(list(set(train_object_list)-set(test_object_list)))) & (df['Pose'].isin(train_poses)) & (df['Experiment'].isin(train_experiments)), df.columns.difference(['Slip'], sort=False)]
        # Y_train = df.loc[(df['Object'].isin(list(set(train_object_list)-set(test_object_list)))) & (df['Pose'].isin(train_poses)) & (df['Experiment'].isin(train_experiments)), 'Slip']
        print(" Adding Objects {}, and Exp {}: X_train shape is {}".format(list(set(train_object_list)-set(test_object_list)), train_experiments, X_train.shape))

        for objects in test_object_list:
            if objects in train_object_list:
                # X_train = X_train.append(df.loc[(df['Object'] == objects) & (df['Pose'].isin(list(set(train_poses)-set(test_poses)))) & (df['Experiment'].isin(list(set(train_experiments)-set(test_experiments)))), df.columns.difference(['Slip'], sort=False)])
                X_train = X_train.append(df.loc[(df['Object'] == objects) & (df['Experiment'].isin(list(set(train_experiments)-set(test_experiments))))])
                # Y_train = Y_train.append(df.loc[(df['Object'] == objects) & (df['Pose'].isin(list(set(train_poses)-set(test_poses)))) & (df['Experiment'].isin(list(set(train_experiments)-set(test_experiments)))), 'Slip'])
                print(" Adding Objects {}, and Exp {}: X_train shape is {}".format(objects, list(set(train_experiments)-set(test_experiments)), X_train.shape))
                
                # for pose in test_poses:
                #     if pose in train_poses:
                #         # X_train = X_train.append(df.loc[(df['Object'] == objects) & (df['Pose'] == pose) & (df['Experiment'].isin(list(set(train_experiments)-set(test_experiments)))), df.columns.difference(['Slip'], sort=False)])
                #         X_train = X_train.append(df.loc[(df['Object'] == objects) & (df['Pose'] == pose) & (df['Experiment'].isin(list(set(train_experiments)-set(test_experiments))))])
                #         # Y_train = Y_train.append(df.loc[(df['Object'] == objects) & (df['Pose'] == pose) & (df['Experiment'].isin(list(set(train_experiments)-set(test_experiments)))), 'Slip'])
                #         print(" Adding Objects {}, Poses {} and Exp {}: X_train shape is {}".format(objects, pose, list(set(train_experiments)-set(test_experiments)), X_train.shape))

                for exp in test_experiments:
                    if exp in train_experiments:
                        # X_train = X_train.append(df.loc[(df['Object'] == objects) & (df['Pose'].isin(list(set(train_poses)-set(test_poses)))) & (df['Experiment'] == exp), df.columns.difference(['Slip'], sort=False)])
                        X_train = X_train.append(df.loc[(df['Object'] == objects) & (df['Experiment'] == exp)])
                        # Y_train = Y_train.append(df.loc[(df['Object'] == objects) & (df['Pose'].isin(list(set(train_poses)-set(test_poses)))) & (df['Experiment'] == exp), 'Slip'])
                        print(" Adding Objects {}, and Exp {}: X_train shape is {}".format(objects, exp, X_train.shape))

        X_train, X_val, Y_train, Y_val = train_test_split(X_train.loc[:,X_train.columns.difference(['Age'], sort=False)], X_train['Age'], test_size=0.33, stratify=X_train['Age'], random_state=123456)

        print('Train data distribution')
        load_exp.tactile_label_statistics(Y_train)
        load_exp.get_experiment_tactile_labels_statistics([X_train['Object'], X_train['Experiment']])
        print('Validation data distribution')
        load_exp.tactile_label_statistics(Y_val)
        load_exp.get_experiment_tactile_labels_statistics([X_val['Object'], X_val['Experiment']])

        
        print("##########Building test dataset....... ")
        X_test = df.loc[(df['Object'].isin(test_object_list)) & (df['Experiment'].isin(test_experiments)), df.columns.difference(['Age'], sort=False)]
        # X_test = df.loc[(df['Object'].isin(test_object_list)) & (df['Pose'].isin(test_poses)) & (df['Experiment'].isin(test_experiments)), df.columns.difference(['Object','Slip', 'Pose', 'Experiment'], sort=False)]
        Y_test = df.loc[(df['Object'].isin(test_object_list)) & (df['Experiment'].isin(test_experiments)), 'Age']
        print(" Adding Objects {}, and Exp {}: X_test shape is {}".format(test_object_list, test_experiments, X_test.shape))

        print('Test data distribution')
        load_exp.tactile_label_statistics(Y_test)
        load_exp.get_experiment_tactile_labels_statistics([X_test['Object'], X_test['Experiment']])


        X_train = X_train.loc[:,X_train.columns.difference(['Object', 'Experiment','Timestamps', '0_X_cal', '1_X_cal', '2_X_cal', '3_X_cal', '0_Y_cal', '1_Y_cal', '2_Y_cal', '3_Y_cal', '0_Z_cal', '1_Z_cal', '2_Z_cal', '3_Z_cal'])]
        X_val = X_val.loc[:,X_val.columns.difference(['Object', 'Experiment','Timestamps', '0_X_cal', '1_X_cal', '2_X_cal', '3_X_cal', '0_Y_cal', '1_Y_cal', '2_Y_cal', '3_Y_cal', '0_Z_cal', '1_Z_cal', '2_Z_cal', '3_Z_cal'])]
        X_test = X_test.loc[:,X_test.columns.difference(['Object', 'Experiment','Timestamps', '0_X_cal', '1_X_cal', '2_X_cal', '3_X_cal', '0_Y_cal', '1_Y_cal', '2_Y_cal', '3_Y_cal', '0_Z_cal', '1_Z_cal', '2_Z_cal', '3_Z_cal'])]
        
        if use_pca:
            ## PCA
            scaler = StandardScaler()# Fit on training set only.
            scaler.fit(X_train)# Apply transform to both the training set and the test set.
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Make an instance of the Model
            pca = PCA(.95)
            pca.fit(X_train)

            X_train = pca.transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)

            print('Using PCA. Number of choosen components is {}'.format(pca.n_components_))
            global max_principal_components
            max_principal_components = pca.n_components_
        
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, rf_max_depth, rf_decision_trees, classification_prob_thereshold


    return None
    # return df

def ml_model_predict(ml_name, dataset_str, ml_model, x, y, exp_index):
    global f1_scores_test
    global f1_scores_val
    print("Classification using model {}; for dataset {} and exp_index {}".format(ml_name, dataset_str, exp_index))

    predicted = ml_model.predict(x)

    accuracy = accuracy_score(y, predicted)
    precision = precision_score(y, predicted)
    recall = recall_score(y, predicted)
    fscore = f1_score(y, predicted)

    if dataset_str == "test":
        f1_scores_test[ml_name][exp_index] = fscore
        print(" --> f1_score for test dataset was: {}".format(f1_scores_test[ml_name]))

    if dataset_str == "validation":
        f1_scores_val[ml_name][exp_index] = fscore
        print(" --> f1_score for validation dataset was: {}".format(f1_scores_val[ml_name]))

    # Saving results
    f = open("results/"+ml_name+"/"+dataset_str+"/"+dataset_str+"_"+ml_name+"_"+str(exp_index)+"_results.txt", "w")
    if not use_pca:
        f.write('Training Features {} \n'.format(x.columns.array))
    else:
        f.write('Features are first {} PCA principal components \n'.format(max_principal_components))
    # f.write('Training data with shape {} \n'.format(X_train.shape))
    # labelStats(f, Y_train)
    f.write('Predicting dataset with shape {} \n'.format(x.shape))
    # labelStats(f, Y_test)

    # f.write('\nRF params - max_depth: {}, decision_trees: {} \n'.format(rf_max_depth, rf_decision_trees))
    # f.write('ROC curve analysis suggests threshold with value {}\n'.format(thresholds[index_aux]))
    # f.write('Using classification probability thereshold {}\n'.format(classification_prob_thereshold))
    # f.write('Out-of-bag score estimate: {}\n'.format(ml_model.oob_score_))
    f.write('Mean accuracy score: {}\n'.format(accuracy))
    f.write('Mean precision score: {}\n'.format(precision))
    f.write('Mean recall score: {}\n'.format(recall))
    f.write('F1 score is: {}\n'.format(fscore))
    f.write('Confusion matrix is {}\n'.format(confusion_matrix(y, predicted)))
    
    # f.write('\n\nFeature importance: {}\n'.format(fi))

    f.close()

    return predicted


if __name__ == '__main__':

    global experiment_type

    for file_num in range(number_of_trials):
    # for file_num in [6]:
        if experiment_type == 'val':
            # X_train and X_test are empty lists
            X_train, X_val, X_test, Y_train, Y_val, Y_test,rf_max_depth, rf_decision_trees, classification_prob_thereshold = load_train_validation_data('../config/slip_model_training_data.json')

        elif experiment_type == 'val_and_test':
            X_train, X_val, X_test, Y_train, Y_val, Y_test, rf_max_depth, rf_decision_trees, classification_prob_thereshold = load_train_validation_data('../config/slip_model_training_data_'+str(file_num)+'.json')


        for ml_name in ml_models:
        # for ml_name in ["SVM", "Bayes", "KNN"]:
            if ml_name == "random_forest":
                # Model training
                rf = RandomForestClassifier(max_depth = rf_max_depth, n_estimators=rf_decision_trees, oob_score=True, 
                                            random_state=123456, 
                                            class_weight= "balanced", 
                                            n_jobs=-1)
                
                rf.fit(X_train, Y_train)

                ## Save learnt model externaly  
                # dump(rf, 'constrained_data_models/slip_rf.joblib', protocol=2)
                
                # fi = pd.DataFrame({'feature': [keys for keys in X_train.columns if keys != 'Object_Slip'],
                #             'importance': rf.feature_importances_}).\
                #                 sort_values('importance', ascending = False)

                # print('Out-of-bag score estimate: {}\n'.format(rf.oob_score_))
                

                ml_model = rf
                
            elif ml_name == "SVM":
                svm_poly = SVC(kernel="poly")

                svm_poly.fit(X_train, Y_train)
                ## Save learnt model externaly  
                # dump(svm_poly, 'constrained_data_models/slip_svm.joblib', protocol=2)

                ml_model = svm_poly

            elif ml_name == "GaussianProcess":
                my_rbf_gp = GaussianProcessClassifier(RBF())

                my_rbf_gp.fit(X_train, Y_train)
                ## Save learnt model externaly  
                # dump(my_rbf_gp, 'constrained_data_models/slip_gaussian.joblib', protocol=2)

                ml_model = my_rbf_gp

            elif ml_name == "AdaBoost":
                my_adaboost = AdaBoostClassifier(n_estimators=300, learning_rate=0.1)
                my_adaboost.fit(X_train, Y_train)
                ## Save learnt model externaly  
                # dump(my_adaboost, 'constrained_data_models/slip_adaboost.joblib', protocol=2)

                ml_model = my_adaboost

            elif ml_name == "MLP_default":
                mlp_default = MLPClassifier()

                mlp_default.fit(X_train, Y_train)
                ## Save learnt model externaly  
                # dump(mlp_default, 'constrained_data_models/slip_mlp_default.joblib', protocol=2)

                ml_model = mlp_default

            elif ml_name == "MLP_1":
                mlp_1 = MLPClassifier(hidden_layer_sizes=(128), activation="logistic", learning_rate_init=0.0005)
                            
                mlp_1.fit(X_train, Y_train)
                ## Save learnt model externaly  
                # dump(mlp_1, 'constrained_data_models/slip_mlp_1.joblib', protocol=2)

                ml_model = mlp_1

            elif ml_name == "MLP_2":
                mlp_2 = MLPClassifier(hidden_layer_sizes=(256), activation="logistic", learning_rate_init=0.0005)

                mlp_2.fit(X_train, Y_train)

                ## Save learnt model externaly  
                # dump(mlp_2, 'constrained_data_models/slip_mlp_2.joblib', protocol=2)

                ml_model = mlp_2        
        
            elif ml_name == "MLP_3":
                mlp_3 = MLPClassifier(hidden_layer_sizes=(512), activation="logistic", learning_rate_init=0.0005)

                mlp_3.fit(X_train, Y_train)

                ## Save learnt model externaly  
                # dump(mlp_3, 'constrained_data_models/slip_mlp_3.joblib', protocol=2)

                ml_model = mlp_3

        
            elif ml_name == "MLP_4":
                mlp_4 = MLPClassifier(hidden_layer_sizes=(512, 256), activation="logistic", learning_rate_init=0.0005)

                mlp_4.fit(X_train, Y_train)

                ## Save learnt model externaly  
                # dump(mlp_4, 'constrained_data_models/slip_mlp_4.joblib', protocol=2)

                ml_model = mlp_4

            elif ml_name == "Bayes":
                naive_default = GaussianNB()
                
                naive_default.fit(X_train, Y_train)

                ## Save learnt model externaly  
                # dump(naive_default, 'constrained_data_models/slip_naive_default.joblib', protocol=2)

                ml_model = naive_default

            elif ml_name == "KNN":
                neigh = KNeighborsClassifier()

                neigh.fit(X_train, Y_train)

                ## Save learnt model externaly  
                # dump(neigh, 'constrained_data_models/slip_neigh.joblib', protocol=2)

                ml_model = neigh
        
            
            for (x_dataset, y_dataset, dataset_name) in zip([X_val, X_test],[Y_val, Y_test],["validation", "test"]):

                plt.clf()

                predicted = ml_model_predict(ml_name, dataset_name, ml_model, x_dataset, y_dataset, file_num)
                # Confusion Matrix
                cm = pd.DataFrame(confusion_matrix(y_dataset, predicted), columns=['Old', 'New'], index=['Old', 'New'])
                # cm = pd.DataFrame(confusion_matrix(Y_test, predicted, normalize='true'), columns=['NO Slip', 'Slip'], index=['NO Slip', 'Slip'])
                
                svm = sns.heatmap(cm, annot=True, cmap="gray_r", cbar=False, vmin=0.0, vmax=1.0)
                        
                    
                plt.savefig("results/"+ml_name+"/"+dataset_name+"/"+dataset_name+"_"+ml_name+"_"+str(file_num)+"_conf_matrix.svg")
                plt.savefig("results/"+ml_name+"/"+dataset_name+"/"+dataset_name+"_"+ml_name+"_"+str(file_num)+"_conf_matrix.png")
                
                plt.clf()

                if experiment_type == 'val': # Only if testing dataset is available
                    break


    ## Compute histograms for validation and testing sets
    for (CTEs, error, f1_scores, dataset_name) in zip([CTEs_val, CTEs_test],[error_val, error_test],[f1_scores_val, f1_scores_test],["validation", "test"]):
        for ml_model, index in zip(ml_models, range(len(ml_models))):
            CTEs[index] = np.mean(f1_scores[ml_model])
            error[index] = np.std(f1_scores[ml_model])
        
        x_pos = np.arange(len(ml_models))

        plt.clf()

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('F1-Score(%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ml_models)
        ax.set_title('F1-Score of different ML models for '+dataset_name+' set')
        ax.yaxis.grid(True)

        ax.tick_params(axis='x', which='major', labelsize=10)
        fig.autofmt_xdate(rotation=55)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('results/'+'overall_f1_score_'+dataset_name+'_set.png')
        plt.savefig('results/'+'overall_f1_score_'+dataset_name+'_set.svg')
        plt.show()
        
        if experiment_type == 'val': # Only if testing dataset is available
            break
