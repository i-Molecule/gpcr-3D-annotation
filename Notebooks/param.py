import os,sys

his_types = ["HSD", "HSE", "HSP"]

path_to_GPCRapa = "/home/ilya/work/Projects/gpcr-3D-annotation/"

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

   
path_to_gpcrdb_files = os.path.join(path_to_GPCRapa,"Files/resources/Gpcrdb_files/")


new_seq_aligned_to_GPCRdb_seq_database = "ns_aligned_to_db.fasta"
canonical_residues = ["1x50", "2x50",
                      "3x50", "4x50",
                      "5x50", "6x50",
                      "7x50", "8x50"]

canonical_residues_dict = {"1x50":"N", "2x50":"DN",
                           "3x50":"R", "4x50":"W",
                           "5x50":"PLV", "6x50":"P",
                           "7x50":"P", "8x50":"FLMV"}


gpcrdb_alignment = "GPCRdb_alignment_honly.fasta"
gpcrdb_numeration = "gpcrdb_numbers_honly.csv"

one_mod_df = os.path.join(path_to_GPCRapa,"Files/resources/mod_data/onemodal_features_dist_stats.csv")
bi_mod_df = os.path.join(path_to_GPCRapa,"Files/resources/mod_data/bimodal_features_dist_stats.csv")
inv_d = {v: k for k, v in d.items()}
one_mod_feat = ["2x42:4x45", "2x43:7x53",
                "2x45:4x50", "3x50:6x37",
                "3x51:5x57", "6x44:6x48",
                "6x44:7x45", "6x48:7x45",
                "7x45:7x49"]
res_contact_list = ["1x49:7x50", "1x53:7x53",
                    "1x53:7x54", "2x37:2x40",
                    "2x42:4x45", "2x43:7x53",
                    "2x45:4x50", "2x46:2x50",
                    "2x50:3x39", "2x50:7x49",
                    "2x57:7x42", "3x40:6x48",
                    "3x43:6x40", "3x43:6x41",
                    "3x43:7x49", "3x43:7x53",
                    "3x46:6x37", "3x46:7x53",
                    "3x46:3x50", "3x49:3x50",
                    "3x50:3x53", "3x50:6x37",
                    "3x50:7x53", "3x51:5x57",
                    "5x51:6x44", "5x55:6x41",
                    "5x58:6x40", "5x62:6x37",
                    "6x40:7x49", "6x44:6x48",
                    "6x44:7x45","6x48:7x45",
                    "7x45:7x49", "7x50:7x55",
                    "7x52:7x53", "7x53:8x50",
                    "7x54:8x50", "7x54:8x51"]
                    
                    
model = os.path.join(path_to_GPCRapa,"Files/resources/RF_classifier_good_model_19.joblib")
