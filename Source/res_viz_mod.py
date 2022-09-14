import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



import df_for_trj_mod as dfftrj
#resuts visualization
def traj_viz(pdb_id:str, path_to_gpcrdb_files:str, dir_path:str, out_path:str,
             gpcrdb_alignment:str, gpcrdb_numeration:str, one_mod_df:str,
             bi_mod_df:str, model:str, frames_dir_path:str, inv_d:dict,
             res_contact_list:list, one_mod_feat:list,
             new_seq_aligned_to_GPCRdb_seq_database:str, canonical_residues_dict:dict,
             his_types:list, d:dict
            ):
    
    
    """Visualizes the RF model classification result through all the trajectory frames.  

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
      
      model:
          Path to the RF model.
      
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
    
      Graph representation of the RF model classification on all trajectory pdb files in order.
      Dots represent the active probability score from the corresponding pdb files. The curve
      represents smoothed data, which is obtained by taking the mean value of active probability
      score from five consecutive MD frames. Red line represents the classification threshold,
      the dots above it are classified as active structures, bellow it are classified as inactive.
    """
    
    
    plt.figure(figsize = (30, 10))
    classifier_values_df = dfftrj.trajectory_df_for_viz(pdb_id, path_to_gpcrdb_files, dir_path, out_path,
                                                 	  gpcrdb_alignment, gpcrdb_numeration, one_mod_df,
                                                 	  bi_mod_df, model, frames_dir_path, inv_d,
                                                 	  res_contact_list, one_mod_feat, 
                                                 	  new_seq_aligned_to_GPCRdb_seq_database, 
                                                 	  canonical_residues_dict, his_types, d)
    
    
    ap_values = classifier_values_df["Active probability"].to_list()
    lim = len(ap_values)
    ax1 = sns.scatterplot(data=classifier_values_df,
                          x="frame",
                          y="Active probability",
                          alpha = 0.2,
                          palette = "flare").set(ylim=(0, 1), xlim=(0, lim))
    
    #curve smoothing
    smoothed_values_dfs = []
    ap_values_split = np.array_split(ap_values, len(ap_values)//5)
    mean_ap_values = [f.mean() for f in ap_values_split]
    mean_classifier_values_df = pd.DataFrame(mean_ap_values, columns = ['mean_ap_values'])
    mean_classifier_values_df["frame"] = [f*5 for f in range(0, len(ap_values)//5)]
    #smoothed_values_dfs.append(mean_classifier_values_df)
    #df_s = pd.concat(smoothed_dfs)
    
    
    ax1 = sns.lineplot(data=mean_classifier_values_df,
                       x="frame",
                       y="mean_ap_values",
                       palette = "flare").set(ylim=(0, 1), xlim=(0, lim))
    
    
    plt.axhline(y=0.375, c="red")
    
    
    return ax1
