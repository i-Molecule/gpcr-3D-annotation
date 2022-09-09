import os,sys
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


import mapping_and_seq_mod as ms
import feature_calc_mod as fc

def main_feat_traj(pdb_id:str, path_to_gpcrdb_files:str, dir_path:str, out_path:str,
                   gpcrdb_alignment:str, gpcrdb_numeration:str, one_mod_df:str,
                   bi_mod_df:str, frames_dir_path:str, inv_d:dict,
                   res_contact_list:list, one_mod_feat:list,
                   new_seq_aligned_to_GPCRdb_seq_database:str, canonical_residues_dict:dict,
                   his_types:list, d:dict
                   ) -> pd.DataFrame():
    
    """Calculates the array of RF model classification results for the MD trajectory splitted into
       pdb files according to simulation frames.

    Args:
    
      pdb_id:
          Name of pdb file.
          
      path_to_gpcrdb_files:
          Path to resource GPCRdb files.
          
      dir_path:
          Path to save secondary files (sequence files and alignments), 
          which will be created using this function.
          
      out_path:
          Path to save primary file that will be the result of this function.
          
      gpcrdb_alignment:
          Filename of the GPCR class A alignment taken from GPCRdb.
          
      gpcrdb_numeration:
          Filename of numeration file from gpcrdb_alignment.

      one_mod_df:
          Path to the file with information on features with one modality of values distribution.
          
      bi_mod_df:
          Path to the file with information on features with two modalities of values distribution.
      
      frames_dir_path:
          Path to the directory with pdb files, corresponding to frames from the MD trajectory.
          For ....(file name format warning)
          The split of the trajectory file into pdb files, for example, can be performd using
          GROMACS trjconv command on trajectory and toplogy files.
          
      inv_d:
          Dictionary which converts amino acid 1 letter abbreviation to 3 letter abbreviation.
          
      res_contact_list:
          List with interactions between amino acid residues that are crucial for GPCR class A activation.
          
      one_mod_feat:
          List of features with one modality of values distribution.
          
      new_seq_aligned_to_GPCRdb_seq_database:
          The filename of the secondary alignment.
          
      canonical_residues_dict:
          Dictionary with residues that are most common among class A GPCRs for the conserved
          sequence positions in GPCRdb numernation.
    
      his_types:
          List contining the names of different HIS protonation states.
      d:
          Dictionary which converts amino acid 3 letter abbreviation to
            1 letter abbreviation.
            
    Returns:
    
      Classification result of the trajectory pdb files from RF model in np.array data format. The first value corresponds
      to the probability of the analyzed structure being classified as inactive, the second value
      corresponds to the probability of the analyzed structure being classified as active.
    """
    
    
    i=0
    res_arr = []
    PDB_files_sorted_list = sorted([frames_dir_path+f for f in os.listdir(frames_dir_path)],
                                   key=lambda x: int(x.split("/")[-1].rstrip(".pdb").split("_")[1]))
    
    #for filenames like frame_NNN
    for PDBFile in tqdm(PDB_files_sorted_list):
        if i == 0:
            
            map_df = ms.GPCRdb_mapping_for_sequence(pdb_id, PDBFile, path_to_gpcrdb_files, dir_path, out_path,
                                                    new_seq_aligned_to_GPCRdb_seq_database,
                                                    canonical_residues_dict, gpcrdb_alignment,
                                                    gpcrdb_numeration, his_types, d)
            
            res_array = fc.calc_dist_feature_modif_no_c_id(PDBFile, map_df, pdb_id, one_mod_df, bi_mod_df,
                                                           d, res_contact_list, one_mod_feat, his_types)
            

            res_arr.append(res_array)
            i+=1
            
        else:
            
            res_array = fc.calc_dist_feature_modif_no_c_id(PDBFile, map_df, pdb_id, one_mod_df, bi_mod_df,
                                                           d, res_contact_list, one_mod_feat, his_types)
            

            res_arr.append(res_array)
            
    res_arr = pd.concat(res_arr)

            
    return res_arr
