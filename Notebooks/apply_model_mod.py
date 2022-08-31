import pandas as pd
import numpy as np
import joblib

def model_apply(res_df:pd.DataFrame(), model:str) -> np.array:
    
    """Generalizing function uses GPCRdb_mapping_for_sequence and
       calc_dist_feature_modif_no_c_id to calculate features for
       the RF model to obtain its classification result.

    Args:
    
      res_df:
          Dataframe that contains calculated features.(output of calc_dist_feature_modif_no_c_id)
          
      model:
          Path to the RF model.
          
    Returns:
    
      Classification result of single pdb file from RF model in np.array data format. The first value corresponds
      to the probability of the analyzed structure being classified as inactive, the second value
      corresponds to the probability of the analyzed structure being classified as active.
    """
    
    
    res_array = res_df.to_numpy()
    
    
    rf = joblib.load(model)
    res = rf.predict_proba(res_array)
    return res
