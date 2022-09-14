# gpcr-3D-annotation
GPCRapa is a random forest model capable of annotation of GPCR conformations as active or inactive.

## Installation

You can create an environment from .yml file:

```bash
conda env create -f gpcr_3d_conf.yml
conda activate gpcr_3d_conf
```

All required packages and versions can be found in requirements.txt file.

GPCRapa was tested on Ubuntu 20.04.5 LTS, AMD® Ryzen 7 3800x 8-core processor × 16, NVIDIA Corporation TU117 [GeForce GTX 1650].

## Usage

To look at usage example of GPCRapa see Tutorial.ipynb in /Notebooks/

## Training

To train your own model on your own data use script model_train_script.py in /Notebooks/ . It is mandatory that names of the extracted framers should be in "(framename)_(number).pdb" format.

```
python3 model_train_script.py -mt svm -psm /gpcr-3D-annotation/Files/resources/GPCR_state_map.csv -pfl path_to_your_trajectories -dp /gpcr-3D-annotation/test/ -op /gpcr-3D-annotation/test/ -mop /gpcr-3D-annotation/test/
```
All options:

```
usage: model_train_script.py [-h] [-mt {randomforest,svm,xgboost}] [-psm PATH_TO_STATE_MAPPING] [-pfl PATH_TO_FRAMES_LOCATION] [-dp DIR_PATH] [-op OUT_PATH] [-mop MODEL_OUT_PATH]
                             [-psf PATH_TO_SAVE_FEATURES] [-nsplits N_SPLITS]

options:
  -h, --help            show this help message and exit
  -mt {randomforest,svm,xgboost}, --modeltype {randomforest,svm,xgboost}
                        model type that will be trained
  -psm PATH_TO_STATE_MAPPING, --path_to_state_mapping PATH_TO_STATE_MAPPING
                        path to state mapping file for your pdbs
  -pfl PATH_TO_FRAMES_LOCATION, --path_to_frames_location PATH_TO_FRAMES_LOCATION
                        path to the folder with frames folders
  -dp DIR_PATH, --dir_path DIR_PATH
                        path to save sequence of pdb file
  -op OUT_PATH, --out_path OUT_PATH
                        path to save generated mapping of pdb file
  -mop MODEL_OUT_PATH, --model_out_path MODEL_OUT_PATH
                        path to save model
  -psf PATH_TO_SAVE_FEATURES, --path_to_save_features PATH_TO_SAVE_FEATURES
                        path to save features
  -nsplits N_SPLITS, --n_splits N_SPLITS
                        number of splits for KFold CV
```
State file example is located in Files/resources/GPCR_state_map.csv

To use custom splitting function to split your data, use model_train_script_custom_split.py:
```
python3 model_train_script_custom_split.py -mt svm -psm /home/ilya/work/Projects/gpcr-3D-annotation/Files/resources/GPCR_state_map.csv -pfl /home/ilya/work/Finished_md/md_finished_23.02.2020/md_frames_for_train/ -dp /home/ilya/work/Projects/gpcr-3D-annotation/test/ -op /home/ilya/work/Projects/gpcr-3D-annotation/test/ -mop /home/ilya/work/Projects/gpcr-3D-annotation/test/ -psf /home/ilya/work/Projects/gpcr-3D-annotation/test/
```
All options:
```
usage: model_train_script_custom_split.py [-h] [-mt {randomforest,svm,xgboost}] [-psm PATH_TO_STATE_MAPPING] [-pfl PATH_TO_FRAMES_LOCATION] [-dp DIR_PATH] [-op OUT_PATH] [-mop MODEL_OUT_PATH]
                                          [-psf PATH_TO_SAVE_FEATURES]

options:
  -h, --help            show this help message and exit
  -mt {randomforest,svm,xgboost}, --modeltype {randomforest,svm,xgboost}
                        model type that will be trained
  -psm PATH_TO_STATE_MAPPING, --path_to_state_mapping PATH_TO_STATE_MAPPING
                        path to state mapping file for your pdbs
  -pfl PATH_TO_FRAMES_LOCATION, --path_to_frames_location PATH_TO_FRAMES_LOCATION
                        path to the folder with frames folders
  -dp DIR_PATH, --dir_path DIR_PATH
                        path to save sequence of pdb file
  -op OUT_PATH, --out_path OUT_PATH
                        path to save generated mapping of pdb file
  -mop MODEL_OUT_PATH, --model_out_path MODEL_OUT_PATH
                        path to save model
  -psf PATH_TO_SAVE_FEATURES, --path_to_save_features PATH_TO_SAVE_FEATURES
                        path to save features
```



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
