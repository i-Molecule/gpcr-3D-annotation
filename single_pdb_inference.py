"""
Small inference script that can be called via commandline using argparse.
Use the pre-trained model to predict the active and inactive probabilities for a single pdb file.

Required arguments:
-i : path to input .pdb file
-o : path to output directory

Optional arguments:
--id : id of the pdb file

Example usage:
python ./single_pdb_inference.py -i ./Files/examples/Single_pdbs/2YDV_frame_22_act.pdb -o ./example_output
"""

import argparse
import pathlib
import sys
import os

# Relative import of the source code, assumes that the script is in the main directory. (gpcr-3D-annotation)
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "Source"))

# import GPCRapa modules
import param as pr  # parameters file
import mapping_and_seq_mod as ms  # module used for deriving sequence from pdb file and making mapping for gpcrdb numeration
import feature_calc_mod as fc  # module that is used for feature calculation for pdb file
import apply_model_mod as aplm  # module that is used to apply trained model(inference)


def apply_aplm_model(pdb_p: pathlib.Path, out_dir: pathlib.Path, pdb_id: str, verbose: bool = False):
    """Get the active and inactive probabilities for a given pdb file.

    Args:
        pdb_id: str
            Identifier for the pdb file. Set to "" to use the pdb file name.
        pdb_p: pathlib.Path
            Path to the pdb file.
        out_dir: pathlib.Path
            Path to the output directory. Will create the directory if it does not exist.
            Creates the mapping file and feature file in this directory.
            Will save a .csv file with the probabilities in this directory.

    Returns:
        Tuple[float, float]: inactive_prob, active_prob
            The probabilities of the structure being inactive and active, respectively.
    """
    # check args
    if not pdb_p.exists():
        raise FileNotFoundError(f"File {pdb_p} does not exist.")
    out_dir.mkdir(exist_ok=True)
    if not out_dir.is_dir():
        raise NotADirectoryError(f"{out_dir} is not a directory.")
    if pdb_id == "":
        pdb_id = pdb_p.stem

    # resolve the paths
    pdb_p = pdb_p.resolve()
    out_dir = out_dir.resolve()


    # change directory to out_dir so by default the output files are saved there
    # cd into the directory so the output files are saved there
    print(f"Changing directory to {out_dir}")
    os.chdir(out_dir)

    

    csv_with_probs_p = out_dir / f"{pdb_id}_probabilities.csv"

    # To map the sequence from pdb file to GPCRdb numeration, you need to create mapping file
    map_df = ms.GPCRdb_mapping_for_sequence(
        pdb_id,
        str(pdb_p),
        pr.path_to_gpcrdb_files,
        str(out_dir),
        str(out_dir),
        pr.new_seq_aligned_to_GPCRdb_seq_database,
        pr.canonical_residues_dict,
        pr.gpcrdb_alignment,
        pr.gpcrdb_numeration,
        pr.his_types,
        pr.d,
    )

    # To calculate features you need to use calc_dist_feature_modif_no_c_id function from feature_calc_mod.
    res_df = fc.calc_dist_feature_modif_no_c_id(
        str(pdb_p),
        map_df,
        str(pdb_id),
        pr.one_mod_df,
        pr.bi_mod_df,
        pr.d,
        pr.res_contact_list,
        pr.one_mod_feat,
        pr.his_types,
    )

    # This function returns the array with two values: first is the probability that structure is inactive, second is the probability that structure is active.
    inactive_prob, active_prob = aplm.model_apply(res_df, pr.model)[0]
    out = aplm.model_apply(res_df, pr.model)
    if verbose:
        print(out)
        print(f"Probability of inactive state: {inactive_prob}")
        print(f"Probability of active state: {active_prob}")

    # save the probabilities to a csv file
    csv_with_probs_p.write_text(f"inactive_prob,active_prob\n{inactive_prob},{active_prob}\n")
    
    return inactive_prob, active_prob


def run_main_example():
    # hardcoded
    script_dir = pathlib.Path(__file__).resolve().parent
    pdb_id = "tttesttt"
    pdb_file = script_dir / "Files/examples/Single_pdbs/2YDV_frame_22_act.pdb"
    out_dir = script_dir / "example_output"
    out_dir.mkdir(exist_ok=True)
    inactive_prob, active_prob = apply_aplm_model(pdb_file, out_dir, pdb_id)


if __name__ == "__main__":
    # For testing purposes, run the main example
    # run_main_example()

    # parse arguments
    PARSER = argparse.ArgumentParser(description="Inference script for GPCRapa.")
    PARSER.add_argument("-i", "--input", help="Path to input .pdb file", required=True, type=pathlib.Path)
    PARSER.add_argument("-o", "--output", help="Path to output directory", required=True, type=pathlib.Path)
    PARSER.add_argument("--id", help="id of the pdb file", default="", required=False, type=str)
    PARSER.add_argument("--verbose", help="Print more information", action="store_true", required=False)

    ARGS = PARSER.parse_args()
    apply_aplm_model(pdb_p=ARGS.input, out_dir=ARGS.output, 
                     pdb_id=ARGS.id, verbose=ARGS.verbose)