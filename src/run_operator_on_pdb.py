#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7


from __future__ import print_function

# example to run the script: 
# python perform_folding_at_pdbs.py 1e0m unfolded

#import pyrosetta as rosetta
#import tensorflow as tf

#import configparser
import optparse    # for option sorting
import random
import pandas as pd
import os
import glob
import sys
sys.path.append("/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta")
sys.path.append("/opt/cesga/easybuild-cesga/software/MPI/gcc/6.4.0/openmpi/2.1.1/mpi4py/3.0.2-python-3.7.0/lib/python3.7/site-packages")

import pyrosetta
import pyrosetta.rosetta as rosetta

from pathlib import Path

from mpi_rosetta import *

from pyrosetta import (
    init, pose_from_sequence, pose_from_file, Pose, MoveMap, create_score_function, get_fa_scorefxn,
    MonteCarlo, TrialMover, SwitchResidueTypeSetMover, PyJobDistributor,
)

from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.core.scoring import CA_rmsd

from structure_reader import StructureReader, StraightMover
from structure_scoring import PoseScoring, NeuralNetworkFunction

init(extra_options = "-mute all")  # WARNING: option '-constant_seed' is for testing only! MAKE SURE TO REMOVE IT IN PRODUCTION RUNS!!!!!

native_path = {
    "1e0m": "./data_pdbs/1E0M/1e0m_A.pdb",
    "5wod": "./data_pdbs/5WOD/5wod_A.pdb",
    "1d5q": "./data_pdbs/1D5Q/1d5q.pdb"
}


def get_weights_operator(input_operator):
    weights_operator = []
    with open(input_operator, "r") as inputs:
        lines = inputs.readlines()[1:]
        lines = [float(l.strip()) for l in lines]
        weights_operator = lines
    if len(weights_operator) < 1:
        print("problem reading weights of ann operator")
        exit()
    return weights_operator



def init_basics(prot):
    reader = StructureReader()
    # native protein
    native = reader.get_from_file(native_path[prot])
    straightMover = StraightMover(native)
    # straigh model
    straightPose = straightMover.get_model()
    # pose score
    scorefxn = PoseScoring(native)
    return reader, scorefxn, straightPose, native


def run_operator_on_pdb(input_operator, prot):
    weights_operator = get_weights_operator(input_operator)
       
    reader, scorefxn, straightPose, native = init_basics(prot)
    pdb_files = glob.glob("run_operator/*.pdb")
    print("run with ", len(pdb_files))
    
    # ITERATE THROUGHT INPUT PDBS
    results = []
    # for file_id in pdb_files:
    #     try:
    #         model = reader.get_from_file(file_id)
    #     except:
    #         print("problem with ", file_id)
    #         continue
    #     init_score = scorefxn.score(model)
    #     init_rmsd = CA_rmsd(native, model)
    #     cost_func = NeuralNetworkFunction([model], scorefxn, native)
    #     score, rmsd = cost_func.score_and_rmsd(weights_operator)
    #     results.append({"id": file_id, "init_score": init_score,
    #                 "init_rmsd":init_rmsd, "score":score, "rmsd":rmsd})
    
    straightPose.dump_pdb("final_pose_000I.pdb")
    init_score = scorefxn.score(straightPose)
    init_rmsd = CA_rmsd(native, straightPose)
    cost_func = NeuralNetworkFunction([straightPose], scorefxn, native)
    score, rmsd = cost_func.score_and_rmsd(weights_operator)
    results.append({"id": "straight", "init_score": init_score,
                    "init_rmsd":init_rmsd, "score":score, "rmsd":rmsd})

    print("*** init_score {}".format(init_score))
    print("*** init_rmsd {}".format(init_rmsd))
    print("*** score {}".format(score))
    print("*** rmsd {}".format(rmsd))
    #print(pd.DataFrame(results).head())
    
def main():
    #os.environ["job_id"] = str(random.randint(0,1000))
    os.environ["job_id"] = "1"
    os.environ["output_folder"] = "./" 

    # if input_with_options:
    input_operator = sys.argv[-2]
    prot = sys.argv[-1]
    #input_operator, prot, experiment, source_pdb = read_options_file(sys.argv[-1])
    # else:
    #     # list_bests = sys.argv[-3]
    #     input_operator = sys.argv[-3]
    #     prot = sys.argv[-2]
    #     experiment = sys.argv[-1]
    #     if "Fase2" in rosetta_exp:
    #         rosetta_output = rosetta_output_2Fases
    #     else:
    #         rosetta_output = rosetta_output_Pocos
    #     source_pdb = rosetta_output[prot]

    run_operator_on_pdb(input_operator, prot)
    
if __name__== "__main__":
    main()

