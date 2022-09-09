import os, sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import feature_calc_mod as fc
import mapping_and_seq_mod as ms
import param as pr

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-mt", "--modeltype", type=str, choices=["randomforest", "svm", "xgboost"],
                    help="model type that will be trained", default="randomforest")
parser.add_argument("-psm", "--path_to_state_mapping", type=str,
                    help="path to state mapping file for your pdbs")
parser.add_argument("-pfl", "--path_to_frames_location", type=str,
                    help="path to the folder with frames folders")
parser.add_argument("-dp", "--dir_path", type=str,
                    help="path to save sequence of pdb file")
parser.add_argument("-op", "--out_path", type=str,
                    help="path to save generated mapping of pdb file")
parser.add_argument("-mop", "--model_out_path", type=str,
                    help="path to save model")
args = parser.parse_args()

model_type=args.modeltype#"xgboost"
path_to_state_mapping = args.path_to_state_mapping#"/home/ilya/work/Projects/gpcr-3D-annotation/Files/resources/GPCR_state_map.csv"
path_to_frames_location = args.path_to_frames_location#"/home/ilya/work/Finished_md/md_finished_23.02.2020/md_frames_for_train/"
dir_path = args.dir_path#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"
out_path = args.out_path#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"
model_out_path = args.model_out_path#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"


def custom_split_2_6_wo_repeats(data:pd.DataFrame(),
                                path_to_state_mapping:str,
                                a_train = 2,
                                n_in_str_max = 6,
                                n_in_str_min = 5,
                                n_samples_26 = 3,
                                n_samples_25 = 2,
                                seed = 42) -> list:
    
    
    """Creates custom split for the GPCR class A MD trajectories dataset for 5-fold CV, with the respect
    of the dataset initial proportion. All frames from the trajectory go either in training or test set. 

    Args:
    
      data:
          Dataframe with calculated features of each MD frame of the dataset.
     
      path_to_state_mapping:
      
          Path to the Dataframe that contains mapping for all the dataset structures, particularly their
          state classification.
          
      a_train:
          Number of active structures in training set (Default - 8).
          
      n_in_str_max:
          Maximum number of inactive strucutes in samples for CV (Default - 6).
          
      n_in_str_min:
          Minimum number of inactive strucutes in samples for CV (Default - 5).
          
      n_samples_26:
          Number of samples with 2 active and 6 inactive structures for CV (Default - 3).
          
      n_samples_25:
          Number of of samples with 2 active and 5 inactive structures for CV (Default - 2).
          
      seed:
          Random seed.
          
    Returns:
    
      List, containing the splits for 5-fold CV. Each custom split consists of training and
      test set with labelling.
    """
    import random
    
    
    #making of the mapping dict
    state_mapping_cut = pd.read_csv(path_to_state_mapping)
    state_mapping_dict = {}
    for item1, item2 in zip(state_mapping_cut["PDB code"].tolist(), state_mapping_cut["State"].tolist()):
        state_mapping_dict[item1] = item2
    
    
    active_pdb_list = state_mapping_cut[state_mapping_cut["State"] == "active"]["PDB code"].tolist()
    inactive_pdb_list = state_mapping_cut[state_mapping_cut["State"] == "inactive"]["PDB code"].tolist()
    
    
    #splitting
    i=0
    cv_sets_list = []
    added_pdbs = []
    random.seed(seed)
    while i<n_samples_26:
        
        active_pdb_set = random.sample([f for f in active_pdb_list if f not in added_pdbs], a_train)
        inactive_pdb_set = random.sample([f for f in inactive_pdb_list if f not in added_pdbs], n_in_str_max)
        cv_sets_list.append(active_pdb_set+inactive_pdb_set)
        added_pdbs+=active_pdb_set+inactive_pdb_set
        i+=1
    
    i=0
    while i<n_samples_25:
        
        active_pdb_set = random.sample([f for f in active_pdb_list if f not in added_pdbs], a_train)
        inactive_pdb_set = random.sample([f for f in inactive_pdb_list if f not in added_pdbs], n_in_str_min)
        cv_sets_list.append(active_pdb_set+inactive_pdb_set)
        added_pdbs+=active_pdb_set+inactive_pdb_set
        i+=1
    
    print(cv_sets_list)
    print(len(list(set(added_pdbs))))
    
    
    #data processing 
    data["Pdb_names"] = [f.split("_")[0] for f in feat_df.index]
    cl = [f for f in data.columns.tolist() if f not in ["Unnamed: 0", "Pdb_names"]]
    
    #CV sets preparation
    cv_sets=[]
    for test_set in cv_sets_list:
        train_sets=[f for f in cv_sets_list if f != test_set]
        
        
        #data for k-1 train
        train_dfs=[]
        for train_set in train_sets:
            train_df = data[data["Pdb_names"].isin(train_set)].copy()
            train_dfs.append(train_df)
        train_df = pd.concat(train_dfs)
        print(train_df.shape)
        
        
        #labels for k-1 train
        train_labels=[]
        for item in train_df.index.tolist():
            if state_mapping_dict[item.split("_")[0]] == "active":
                train_labels.append(1)
            elif state_mapping_dict[item.split("_")[0]] == "inactive":
                train_labels.append(0)
        
        train_labels = np.array(train_labels)
        train_array = train_df[cl].to_numpy()
        
        
        #data for test
        test_df = data[data["Pdb_names"].isin(test_set)].copy()
        print(test_df.shape)

        
        #labels for test
        test_labels=[]
        for item in test_df.index.tolist():
            if state_mapping_dict[item.split("_")[0]] == "active":
                test_labels.append(1)
            elif state_mapping_dict[item.split("_")[0]] == "inactive":
                test_labels.append(0)
        
        test_labels = np.array(test_labels)
        test_array = test_df[cl].to_numpy()
        
        
        cv_sets.append([[test_labels,test_array],[train_labels,train_array]])
        
        
    return cv_sets

#calculate features

feat_df = []
pdb_list = []
for folder in tqdm(os.listdir(path_to_frames_location)):
    
    
        frames_dir_path = path_to_frames_location+folder+"/"
        pdb_id = folder
        map_df = ms.GPCRdb_mapping_for_sequence(pdb_id,
                                                frames_dir_path+"frame_0.pdb",
                                                pr.path_to_gpcrdb_files,
                                                dir_path, out_path,
                                                pr.new_seq_aligned_to_GPCRdb_seq_database,
                                                pr.canonical_residues_dict,
                                                pr.gpcrdb_alignment,
                                                pr.gpcrdb_numeration,
                                                pr.his_types, pr.d)
        feats_for_model = []
        pdb_list.append(pdb_id)
        #special case 'ClassA_ednrb_human_5XPR_refined_Inactive_2020-10-08_GPCRdb'
        if folder == '5XPR':
            map_df["GPCRdb_numeration"][48] = '2x37'

        for frame in os.listdir(frames_dir_path):

            frame = frames_dir_path+frame
            res_df = fc.calc_dist_feature_modif_no_c_id(frame,
                                                        map_df,
                                                        pdb_id,
                                                        pr.one_mod_df,
                                                        pr.bi_mod_df,
                                                        pr.d,
                                                        pr.res_contact_list,
                                                        pr.one_mod_feat,
                                                        pr.his_types)
            feats_for_model.append(res_df)

        feats_for_model = pd.concat(feats_for_model)
        feat_df.append(feats_for_model)
    

        
feat_df = pd.concat(feat_df)

    
if len(bad) != 0:
    raise ValueError("Smth went wrong with folders:", bad)
    
else:
    print("Features were calculated")
    

#features split
cv_data_wo_rep_26 = custom_split_2_6_wo_repeats(feat_df,
                                                path_to_state_mapping,
                                                a_train = 2,
                                                n_in_str_max = 6,
                                                n_in_str_min = 5,
                                                n_samples_26 = 3,
                                                n_samples_25 = 2,
                                                seed = 42)

print("Features were splitted for CV")

if model_type == "randomforest":
    
    
    from sklearn.ensemble import RandomForestClassifier
    cv_results = []
    for set_ in cv_data_wo_rep_26:


        clf = RandomForestClassifier(n_estimators=100)


        test_set = set_[0]
        test_labels = test_set[0]
        test_array = test_set[1]


        train_set = set_[1]
        train_labels = train_set[0]
        train_array = train_set[1]

        clf.fit(train_array, train_labels)
        cv_results.append(clf.score(test_array, test_labels))


    print(cv_results, np.mean(cv_results), np.std(cv_results))
    
    
    joblib.dump(clf, model_out_path+'gpcrapa_rf_model.joblib')
    print("Path to random forest model:", model_out_path+'gpcrapa_rf_model.joblib')
    
    
elif model_type == "svm":
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    cv_results = []
    for set_ in cv_data_wo_rep_26:
    
    
        clf_svm = svm.SVC(kernel='rbf', probability=True)


        test_set = set_[0]
        test_labels = test_set[0]
        test_array = test_set[1]


        train_set = set_[1]
        train_labels = train_set[0]
        train_array = train_set[1]


        scaler = StandardScaler()
        train_array = scaler.fit_transform(train_array)
        test_array = scaler.transform(test_array)


        clf_svm.fit(train_array, train_labels)
        cv_results.append(clf_svm.score(test_array, test_labels))
    
    
    print(cv_results, np.mean(cv_results), np.std(cv_results))
    
    
    joblib.dump(clf_svm, model_out_path+'gpcrapa_svm_model.joblib')
    joblib.dump(scaler, model_out_path+'gpcrapa_svm_model_scaler.joblib')
    print("Path to svm model:", model_out_path+'gpcrapa_svm_model.joblib')
    print("Path to svm model scaler:", model_out_path+'gpcrapa_svm_model_scaler.joblib')
    
    
elif model_type == "xgboost":
    
    
    from xgboost import XGBClassifier
    cv_results = []
    for set_ in cv_data_wo_rep_26:


        xg_clf = XGBClassifier(n_estimators=100)


        test_set = set_[0]
        test_labels = test_set[0]
        test_array = test_set[1]


        train_set = set_[1]
        train_labels = train_set[0]
        train_array = train_set[1]

        xg_clf.fit(train_array, train_labels)
        cv_results.append(xg_clf.score(test_array, test_labels))


    print(cv_results, np.mean(cv_results), np.std(cv_results))
    
    joblib.dump(xg_clf, model_out_path+'gpcrapa_xgboost_model.joblib') 
    print("Path to xgboost model:", model_out_path+'gpcrapa_xgboost_model.joblib')
    
    
else:
    raise ValueError('Choose a coorrect model_type', 'randomforest', 'svm', 'xgboost')
    
