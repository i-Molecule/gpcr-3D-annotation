import pandas as pd
import os
import warnings

def get_true_seq(PDBFile:str, his_types:list, d:dict) -> str:
    
    
    """Parses and returns aminoacid sequence from the pdbfile.
    
    Args:
    
        PDBFile:
            Path to pdb file.
            
        his_types:
            List contining the names of different HIS protonation states.
            
        d:
            Dictionary which converts amino acid 3 letter abbreviation to
            1 letter abbreviation. 
          
    Returns:
    
        Amino acid sequence in string data format from pdb file.
    """
    
    ## read all lines from pdb file
    with open(PDBFile, 'r') as file:
        all_lines = file.readlines()
    
    
    ## select lines that contain ATOM infromation from pdbfile lines    
    atom_lines = [l for l in all_lines if l[0:6] == 'ATOM  ']
    tot_seq = []
    
    
    ## parce selected atom lines for residue name and number
    for line in atom_lines:
        res_name = line[17:20]
        
        ## if residue is modified histidine residue, change it to ordinary one
        if res_name in his_types:
            res_name = "HIS"
        else:
            pass

        
        res_num = int(line[22:26].strip())
        tot_seq.append(str(res_num)+"_"+res_name)

        
        ## get rid of duplicates
        tot_seq = list(set(tot_seq))
        ## sort the list in the correct order
        tot_seq.sort(key = lambda x : int(x.split("_")[0]))
    
    
    ## format sequence to fasta
    res_seq = ("").join([d[f.split("_")[1]] for f in tot_seq])
    seq_num = [f.split("_")[0] for f in tot_seq]
    
    return res_seq, seq_num

def GPCRdb_mapping_for_sequence(pdb_id:str, PDBFile:str, path_to_gpcrdb_files:str, dir_path:str, out_path:str,
                                new_seq_aligned_to_GPCRdb_seq_database:str, canonical_residues_dict:dict,
                                gpcrdb_alignment:str, gpcrdb_numeration:str, his_types:list, d:dict  
                               ) -> pd.DataFrame():
    
    
    """Creates GPCRdb mapping file for selected pdb file.
    
    Args:
    
      pdb_id:
          Name of pdb file.
          
      PDBFile:
          Path to pdb file.
          
      path_to_gpcrdb_files:
          Path to resource GPCRdb files.
          
      dir_path:
          Path to save secondary files (sequence files and alignments), 
          which will be created using this function.
          
      out_path:
          Path to save primary file that will be the result of this function.
          
      new_seq_aligned_to_GPCRdb_seq_database:
          The filename of the secondary alignment.
          
      canonical_residues_dict:
          Dictionary with residues that are most common among class A GPCRs for the conserved
          sequence positions in GPCRdb numernation.
          
      gpcrdb_alignment:
          Filename of the GPCR class A alignment taken from GPCRdb.
          
      gpcrdb_numeration:
          Filename of numeration file from gpcrdb_alignment.
          
      his_types:
          List contining the names of different HIS protonation states.
      d:
          Dictionary which converts amino acid 3 letter abbreviation to
            1 letter abbreviation.
      
    Returns:
    
      Pandas dataframe which contains mapping from pdb sequence to GPCRdb numeration.
      Columns:
      
       #letter:
           Amino acid residue from pdb sequence.
           
       position in the original sequence:
           Amino acid residue position from the original sequence.
           
       position in the reference alignment:
           Amino acid residue position in the reference alignment with GPCrdb MSA.
           
       original position from pdb file:
           Amino acid residue position from the original pdb file.
           
       GPCRdb_numeration:
           Amino acid residue position in GPCRdb numeration, which is derived from reference alignment.
           
       original_sequence:
           Amino acid residue position in sequence, which is drived from merging
           "original position from pdb file" and "#letter"
       
    Raises:
    
      UserWarning: If amino acid residue is not the canonical one.
    """
    
    
    ## format the inital PDB file sequence for later use #ref get_true_seq
    res_seq, seq_num = get_true_seq(PDBFile, his_types, d)
    
    
    ##write the PDB file sequence in fasta format to perform alignment
    with open(os.path.join(dir_path,"{}_structure_seq.fasta".format(pdb_id)), "w") as m:
        print('>' + pdb_id, file = m)
        print(res_seq, file = m)
    
    
    ##align initial sequence to GPCRdb alignment
    #PP: define all magic constants (mc) (e.g. "GPCRdb_alignment_honly.fasta") in the corresponding upper cell 
    os.system("mafft --addfull {} --mapout {} > {}".format(os.path.join(dir_path,(pdb_id+"_structure_seq.fasta")),
                                                           os.path.join(path_to_gpcrdb_files,gpcrdb_alignment),
                                                           new_seq_aligned_to_GPCRdb_seq_database))
    
    
    ##read the MSA alignment results
    map_seq = pd.read_csv(os.path.join(dir_path,(pdb_id+"_structure_seq.fasta.map")), skiprows=1)
    ## add original numeration to the MSA alignment results
    map_seq["original position from pdb file"] = seq_num
    #PP: mc; split into several commands
    ## read gpcrdb numerration for MSA alignment results
    gpdb_num = pd.read_csv(os.path.join(path_to_gpcrdb_files,gpcrdb_numeration))["Numeration"].tolist()
    ## fill the GPCRdb numeration for every residue that is preseent in MSA alignment results
    pos_ref_align = [gpdb_num[int(f)-1] if f  != ' -' else " -" for f in map_seq[" position in the reference alignment"].tolist()]
    map_seq["GPCRdb_numeration"] = pos_ref_align
    ##format the data in MSA alignment results for output
    map_seq["original_sequence"] = map_seq["# letter"]+map_seq["original position from pdb file"].astype("str")
    map_seq.to_csv(os.path.join(out_path,"{}_mapping.csv".format(pdb_id)))
    
    
    #test for invalid mapping or mutant(N1.50: 98%, D2.50: 90%, R3.50: 95%, W4.50: 97%, P5.50: 78%, P6.50: 99%, P7.50: 88% F8x50)https://pubmed.ncbi.nlm.nih.gov/24304901/
    #conserved aa residues check
    
    ## filter the MSA alignment results for canonical residues
    check_map_df = map_seq[map_seq["GPCRdb_numeration"].isin(canonical_residues_dict.keys())].copy()
    
    
    for aa in canonical_residues_dict.keys():
        if check_map_df[check_map_df["GPCRdb_numeration"] == aa]["# letter"].tolist()[0] not in canonical_residues_dict[aa]:
            # PP aa is a letter, e.g. D, dict value is a string, e.g. DN, so use 'isin', rather than !=
            warnings.warn("The amino acid residue at position {} seems not to be a canonical one ({}). Please re-check the mapping results!".format(aa, canonical_residues_dict[aa])) 
    
    
    #prints path where the mapping file will be saved
    print(os.path.join(out_path, "{}_mapping.csv".format(pdb_id)))
    
    
    return map_seq
