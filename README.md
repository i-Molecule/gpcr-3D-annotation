# gpcr-3D-annotation
GPCRapa is a random forest model capable of annotation of GPCR conformations as active or inactive.

## Installation

There are several required python packages with which GPCRapa works, so we advise to make and activate a virtual environment first:

```bash
conda create -n gpcr_3d_conf python
conda activate gpcr_3d_conf
```
And than install the required packages:

```
conda install pandas
pip install notebook
conda install -c bioconda mafft
conda install -c conda-forge xgboost
python -m pip install scikit-learn==0.24.1
conda install -c conda-forge tqdm
python -m pip install seaborn==0.11.2
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension 

```
or create an environment from .yml file:

```bash
conda env create -f gpcr_3d_conf.yml
```

## Usage
It is possible to run GPCRapa inside python or using scripts.

```python
import param as pr
import mapping_and_seq_mod as ms
import feature_calc_mod as fc
import apply_model_mod as aplm


# create mapping
map_df = ms.GPCRdb_mapping_for_sequence(pdb_id, PDBFile, pr.path_to_gpcrdb_files, dir_path, out_path,
                                pr.new_seq_aligned_to_GPCRdb_seq_database, pr.canonical_residues_dict,
                                pr.gpcrdb_alignment, pr.gpcrdb_numeration, pr.his_types, pr.d)

# calculate features
res_df = fc.calc_dist_feature_modif_no_c_id(PDBFile, map_df, pdb_id, pr.one_mod_df,
                                pr.bi_mod_df, pr.d, pr.res_contact_list, pr.one_mod_feat, 
                                pr.his_types)

# apply the model
aplm.model_apply(res_df, pr.model)
```
For more detailed explanation see Tutorial.ipynb in /Notebooks/

## Training

To train your own model on your own data use script model_train_script.py in /Notebooks/

```
python3 model_train_script.py -mt svm -psm /gpcr-3D-annotation/Files/resources/GPCR_state_map.csv -pfl path_to_your_trajectories -dp /gpcr-3D-annotation/test/ -op //gpcr-3D-annotation/test/ -mop /gpcr-3D-annotation/test/
```
All options:

```
python3 model_train_script.py [-h] [-mt {randomforest,svm,xgboost}] [-psm PATH_TO_STATE_MAPPING] [-pfl PATH_TO_FRAMES_LOCATION] [-dp DIR_PATH] [-op OUT_PATH] [-mop MODEL_OUT_PATH]

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
```
State file example is located in Files/resources/GPCR_state_map.csv

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
