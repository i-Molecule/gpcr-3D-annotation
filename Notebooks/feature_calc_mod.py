import pandas as pd
import numpy as np


def calc_dist_feature_modif_no_c_id(PDBFile:str, map_df:pd.DataFrame(), pdb_id:str, one_mod_df:str,
                                    bi_mod_df:str, d:dict, res_contact_list:list, one_mod_feat:list,
                                    his_types:list
                                   ) -> pd.DataFrame():
    
    
    """Calculates features vectors from single pdb file.

    Args:
    
      PDBFile:
          Path to pdb file.
          
      map_df:
          Mapping dataframe that is created by using GPCRdb_mapping_for_sequence.
          
      pdb_id:
          Name of pdb file.
          
      one_mod_df:
          Path to the file with information on features with one modality of values distribution.
          
      bi_mod_df:
          Path to the file with information on features with two modalities of values distribution.
          
      d:
          Dictionary which converts amino acid 3 letter abbreviation to 1 letter abbreviation.
          
      res_contact_list:
          List with interactions between amino acid residues that are crucial for GPCR class A activation.
          
      one_mod_feat:
          List of features with one modality of values distribution.
          
      his_types:
          List contining the names of different HIS protonation states.
          
    Returns:
    
      Pandas dataframe with calculated distances between amino acid residues crucial for GPCR 
      class A activation. Each column in it contains a calculated distance between residue pairs
      noted in GPCRdb numeration in column names. Dataframe index is the name of pdb_file.
      
    Raises:
    
      UserWarning1: 
          If one of the residues from res_contact_list is not found in mapping file.
      UserWarning2: 
          If distance value between amino acid residues doesnt belong to the distance value
          distributions based on the statistics of our dataset.
    """
    
    
    mapping = map_df
    
    
##create amino acid (aa) residues mapping dictionary,
    map_dict = {}
    for i in range(len(mapping["original_sequence"].dropna())):
        if mapping["GPCRdb_numeration"].tolist()[i] == "-":
            continue
        amino_acid_pos = str(mapping["original_sequence"].dropna().tolist()[i])
        gpcrdb_mapping_pos = mapping["GPCRdb_numeration"].tolist()[i]
        map_dict[amino_acid_pos] = gpcrdb_mapping_pos
        #e.g. map_dict = {'M1': '1x27', 'P2': '1x28', 'I3': '1x29', ... }
    inv_map_dict = {v: k for k, v in map_dict.items()}
    #e.g. inv_map_dict = {'1x27': 'M1', '1x28': 'P2', '1x29': 'I3', .. }


#check residues content according to mapping
    for residue_pair in res_contact_list:
        res_1=residue_pair.split(":")[0]
        res_2=residue_pair.split(":")[1]
        if res_1 not in inv_map_dict.keys():
            warnings.warn("Those features ({}) won't be computed because of missing mapping labels in mapping file. Check and recalculate mapping file or add missing mapping labels manually.".format(res_1))
        if res_2 not in inv_map_dict.keys():
            warnings.warn("Those features ({}) won't be computed because of missing mapping labels in mapping file. Check and recalculate mapping file or add missing mapping labels manually.".format(res_2))


#create translated list of residues involved in contacts 
# make a list with aa positions from the mapping
    trans_res_contact_list =[]
    residues_inv_in_contacts_list = []
    for residue_pair in res_contact_list:
        first_residue_mapped_aa = inv_map_dict[residue_pair.split(":")[0]][0]         #e.g.'1x27'--> 'M'
        first_residue_position_number = inv_map_dict[residue_pair.split(":")[0]][1:]  #e.g.'1x27'--> '1'
        first_residue_mapped = first_residue_mapped_aa+first_residue_position_number
        
        second_residue_mapped_aa = inv_map_dict[residue_pair.split(":")[1]][0]
        second_residue_position_number = inv_map_dict[residue_pair.split(":")[1]][1:]
        second_residue_mapped = second_residue_mapped_aa+second_residue_position_number
        
        
        trans_res_contact_list.append(first_residue_mapped+":"+second_residue_mapped)
        residues_inv_in_contacts_list.append(first_residue_mapped)
        residues_inv_in_contacts_list.append(second_residue_mapped)
    # e.g trans_res_contact_list = ['G23:P285', 'V27:Y288', 'V27:A289', ...]
    residues_inv_in_contacts_list = list(set(residues_inv_in_contacts_list))
    #e.g. residues_inv_in_contacts_list = ['L48', 'D52', 'I60', 'I287', 'F242', ..]
    
    
#split aa residues contacts into tuples, needed for df grooming
    res_contact_tup = [(f.split(":")[0], f.split(":")[1]) for f in res_contact_list]
    
    
#parse pdb file to retrieve atom coordinates information
    fi = open(PDBFile, 'r')
    all_lines = fi.readlines()
    fi.close()
    atom_lines = [l for l in all_lines if l[0:6] == 'ATOM  ']
    dict_coord = {} # dict to store coordinates. dict_coord[res][atom] = (x,y,z,occ)
    #atom_number_to_name_dict = {} # map atom number to atom name, in order to find N, CA, C, O
    for line in atom_lines:
        # retrive info from each atom line
        #atom_num = int(line[6:11].strip())
        atom_name = line[12:16].replace(' ', '')
        res_name = line[17:20]
        
        
        if res_name in his_types:
            res_name = "HIS"
        else:
            pass
        
        
        res_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        #occ = float(line[54:60].strip())
        res = d[res_name]+str(res_num)
        if res in residues_inv_in_contacts_list:
            #atom_number_to_name_dict[atom_num] = atom_name
            if res not in dict_coord:
                dict_coord[res] = {}
            dict_coord[res][atom_name] = (x, y, z)
        # e.g. dict_coord = {{'G23': {'N': (29.08, 42.86, 67.94), 'HN': (28.88, 43.29, 68.82), ... ,
        #                    'N39': {'N': (44.29, 38.12, 56.72), 'HN': (43.59, 38.64, 57.21), ... }
    

#calculate coordinates of the centers of amino-acids' side chains 
    coords_of_the_center_of_res_sidechain= {}
    for residue in residues_inv_in_contacts_list:
        residue_atoms = dict_coord[residue].keys()
        residue_name = residue[0]
        residue_side_chain_coords = []
        for atom in residue_atoms:
            if (residue_name) == "G":
                if atom in ['N', 'C', 'O']:
                        continue
                (ix, iy, iz) = dict_coord[residue][atom]
                residue_side_chain_coords.append([ix, iy, iz])
            else:
                if atom in ['N', 'CA', 'C', 'O']:
                        continue
                (ix, iy, iz) = dict_coord[residue][atom]
                residue_side_chain_coords.append([ix, iy, iz])
        residue_side_chain_coords = np.array(residue_side_chain_coords)
        mean_residue_side_chain_coords = np.mean(residue_side_chain_coords, axis=0)
        coords_of_the_center_of_res_sidechain[residue] = mean_residue_side_chain_coords
        #e.g. coords_of_the_center_of_res_sidechain = {'F242': array([43.47625 , 54.284375, 73.23375 ]), ..., }


#calculate distance between centers of aa residues side-chains        
    inter_df = []
    for residue_pair in trans_res_contact_list:
        residue_1 = residue_pair.split(":")[0]
        residue_2 = residue_pair.split(":")[1]
        residue_1_coords = coords_of_the_center_of_res_sidechain[residue_1]
        residue_2_coords = coords_of_the_center_of_res_sidechain[residue_2]
        distance = np.linalg.norm(residue_1_coords - residue_2_coords)
        inter_df.append({"Res_1":map_dict[(residue_1)],
                         "Res_2":map_dict[(residue_2)],
                         "{:}_{:}".format(pdb_id,PDBFile.split("/")[-1].rstrip(".pdb")): distance})
    
    
# df data check
    inter_df2 = pd.DataFrame(inter_df).drop_duplicates()
    contacts_df = inter_df2[inter_df2[["Res_1", "Res_2"]].apply(tuple, axis=1).isin(res_contact_tup)]
    
    
#df grooming
    contacts_df["Interacting_residues"] = [":".join([k,f]) for k,f in zip(contacts_df["Res_1"].tolist(), contacts_df["Res_2"].tolist())]
    contacts_df = contacts_df[["Interacting_residues", "{:}_{:}".format(pdb_id, PDBFile.split("/")[-1].rstrip(".pdb"))]].T
    contacts_df = contacts_df.rename(columns=contacts_df.iloc[0]).drop(contacts_df.index[0])
    contacts_df = contacts_df[res_contact_list]
    
    
#sigma threshold checking for bimodal features
    one_mod_df = pd.read_csv(one_mod_df)
    bi_mod_df = pd.read_csv(bi_mod_df)
    
    for feature_value, residue_pair in zip(contacts_df.T[contacts_df.index[0]].values, contacts_df.columns):
        #PP: mc
        if residue_pair in one_mod_feat:
            appr_mean = one_mod_df[one_mod_df["Feature"] == residue_pair]["approx_mean"].values[0]
            appr_std = one_mod_df[one_mod_df["Feature"] == residue_pair]["approx_std"].values[0]
            sigma = 1 #  sigma threshold
            while ((feature_value > appr_mean - appr_std*sigma) and (feature_value < appr_mean + appr_std*sigma)) == False:
                sigma+=1
            if sigma<5 and sigma>3:
                warnings.warn("Smth wrong with feature {}. Check the distance between common activation pathway residues visually!".format(residue_pair))
            elif sigma>= 5:
                warnings.warn('Ambiguous {} feature value {}. Try to check structure mapping and structure model!'.format(residue_pair, feature_value))

                
        else:
            appr_mean_1 = bi_mod_df[bi_mod_df["Feature"] == residue_pair]["approx_mean_1"].values[0]
            appr_std_1 = bi_mod_df[bi_mod_df["Feature"] == residue_pair]["approx_std_1"].values[0]
            appr_mean_2 = bi_mod_df[bi_mod_df["Feature"] == residue_pair]["approx_mean_2"].values[0]
            appr_std_2 = bi_mod_df[bi_mod_df["Feature"] == residue_pair]["approx_std_2"].values[0]
            sigma = 1
            while (((feature_value > appr_mean_1 - appr_std_1*sigma) and (feature_value < appr_mean_1 + appr_std_1*sigma)) or ((feature_value > appr_mean_2 - appr_std_2*sigma) and (feature_value < appr_mean_2 + appr_std_2*sigma))) == False:
                 sigma+=1
            if sigma < 5 and sigma>3:
                warnings.warn("Smth wrong with feature {}. Check the distance between common activation pathway residues visually!".format(residue_pair))
            elif sigma>= 5:
                warnings.warn('Ambiguous {} feature value {}. Try to check structure mapping and structure model!'.format(residue_pair, feature_value))
    
    
    return contacts_df
