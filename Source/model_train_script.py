# This script is used to train random forest, xgboost, svm models on structural distance-based features, derived from a set of GPCRs' pdb files. 


import os, sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm


import feature_calc_mod as fc
import mapping_and_seq_mod as ms
import param as pr


#command line arguments
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
parser.add_argument("-psf", "--path_to_save_features", type=str,
                    help="path to save features")
parser.add_argument("-nsplits", "--n_splits", type=int,
                    help="number of splits for KFold CV")
args = parser.parse_args()


#script arguments and their examples
model_type=args.modeltype#"xgboost"
path_to_state_mapping = args.path_to_state_mapping#"/home/ilya/work/Projects/gpcr-3D-annotation/Files/resources/GPCR_state_map.csv"
path_to_frames_location = args.path_to_frames_location#"/home/ilya/work/Finished_md/md_finished_23.02.2020/md_frames_for_train/"
dir_path = args.dir_path#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"
out_path = args.out_path#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"
model_out_path = args.model_out_path#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"
path_to_save_features = args.path_to_save_features#"/home/ilya/work/Projects/gpcr-3D-annotation/test/"
n_splits = args.n_splits#5


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
print("Features have been calculated")

feat_df.to_csv(path_to_save_features+"features_gpcrapa_new_model.csv")
state_mapping_cut = pd.read_csv(path_to_state_mapping)
active_pdb_list = state_mapping_cut[state_mapping_cut["State"] == "active"]["PDB code"].tolist()
inactive_pdb_list = state_mapping_cut[state_mapping_cut["State"] == "inactive"]["PDB code"].tolist()
dataset_array = feat_df.to_numpy()
dataset_labels= np.array([1 if f in active_pdb_list else 0 for f in [f.split("_")[0] for f in feat_df.index]])



from sklearn.model_selection import KFold
kf = KFold(n_splits=n_splits)
cv_results = []
i=0
for train, test in kf.split(dataset_array):
    
    
    train_array, test_array, train_labels, test_labels = dataset_array[train], dataset_array[test], dataset_labels[train], dataset_labels[test]
    
    
    if model_type == "randomforest":
        
        
    
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train_array, train_labels)
        cv_results.append(clf.score(test_array, test_labels))


        joblib.dump(clf, model_out_path+'gpcrapa_rf_model_fold{}.joblib'.format(i))
        print("Path to random forest model:", model_out_path+'gpcrapa_rf_model_fold{}.joblib'.format(i))
        i+=1
        
        if i == n_splits:
            print(cv_results, np.mean(cv_results), np.std(cv_results))
    
    elif model_type == "svm":


        from sklearn.preprocessing import StandardScaler
        from sklearn import svm


        clf_svm = svm.SVC(kernel='rbf', probability=True)


        scaler = StandardScaler()
        train_array = scaler.fit_transform(train_array)
        test_array = scaler.transform(test_array)


        clf_svm.fit(train_array, train_labels)
        cv_results.append(clf_svm.score(test_array, test_labels))


        joblib.dump(clf_svm, model_out_path+'gpcrapa_svm_model_fold{}.joblib'.format(i))
        joblib.dump(scaler, model_out_path+'gpcrapa_svm_model_scaler_fold{}.joblib'.format(i))
        print("Path to svm model:", model_out_path+'gpcrapa_svm_model_fold{}.joblib'.format(i))
        print("Path to svm model scaler:", model_out_path+'gpcrapa_svm_model_scaler_fold{}.joblib'.format(i))
        i+=1
        
        if i == n_splits:
            print(cv_results, np.mean(cv_results), np.std(cv_results))
    
    elif model_type == "xgboost":
    
    
        from xgboost import XGBClassifier

        
        xg_clf = XGBClassifier(n_estimators=100)
        

        xg_clf.fit(train_array, train_labels)
        cv_results.append(xg_clf.score(test_array, test_labels))


        joblib.dump(xg_clf, model_out_path+'gpcrapa_xgboost_model_fold{}.joblib'.format(i)) 
        print("Path to xgboost model:", model_out_path+'gpcrapa_xgboost_model_fold{}.joblib'.format(i))
        i+=1

        
        if i == n_splits:
            print(cv_results, np.mean(cv_results), np.std(cv_results))
            
            
    else:
        raise ValueError('Choose a coorrect model_type', 'randomforest', 'svm', 'xgboost')
