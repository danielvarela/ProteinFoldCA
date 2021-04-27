#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7


from __future__ import print_function

# example to run the script: 
# python perform_folding_at_pdbs.py 1e0m unfolded

#import pyrosetta as rosetta
#import tensorflow as tf

#import configparser
import optparse    # for option sorting
import random
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

rosetta_output_4Fases = {
    "1e0m": "./Rosetta4Fases_PDBS_1e0m/",
    "5wod": "./Rosetta4Fases_PDBS_5wod/"
}


rosetta_output_2Fases = {
    "1e0m": "./Rosetta2Fases_PDBS_1e0m/",
    "5wod": "./Rosetta2Fases_PDBS_5wod/"
}


rosetta_output_Pocos = {
    "1e0m": "./RosettaPOCOS_1e0m/",
    "5wod": "./RosettaPOCOS_5wod/"
}


jobs_1e0m = { 
   "unfolded" : "./results_1e0mA_starts_with_unfolded/ann_job_388.txt",
   "3modelsAndStraight" :  "./buenosResultados_1e0mA_3modelsAndStraight/ann_job_15.txt",
   "3models" : "./buenosResultados_1e0mA_3models/ann_job_500.txt",
} 
jobs_5wod = {
  "unfolded" :"./results_5wod_A_starts_with_unfolded/ann_job_281.txt",
  "3modelsAndStraight" : "./buenosResultados_5wod_3modelsAndStraight/ann_job_586.txt",
  "3models" : "./buenosResultados_5wod_3models/ann_job_212.txt",
}

# estas ejecuciones tenian errores, usaba el mismo modelo (la nativa)
# jobs_1e0m = {
#     "3modelsAndStraight" : "./results_1e0mA_starts_with_3modelsAndStraight/ann_job_903.txt",
#     "unfolded": "./results_1e0mA_starts_with_unfolded/ann_job_388.txt",
#     "3models" : "./results_1e0mA_starts_with_3models/ann_job_719.txt",
# }

# jobs_5wod = {
#     "unfolded" : "./results_5wod_A_starts_with_unfolded/ann_job_281.txt",
#     "3modelsAndStraight" : "./results_5wod_A_starts_with_3modelsAndStraight/ann_job_901.txt",
#     "unfolded" : "./results_5wod_A_starts_with_3models/ann_job_954.txt",
# }


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


def get_pdb_files(prot, source_pdb):
    pdb_files = glob.glob(source_pdb + "/*.pdb")
    pdb_files = [pdb_files[0]]
    if len(pdb_files) < 1:
        print("problem reading rosetta files")
        exit()
    return pdb_files    

def print_results(name, results):
    results_csv = open(name, "w")
    results_csv.write("id,init_score,init_rmsd,score,rmsd\n")
    for r in results:
        results_csv.write("{},{},{},{},{}\n".format(r["id"], r["init_score"], r["init_rmsd"], r["score"], r["rmsd"]))
    


def run(input_operator, prot, experiment, input_source):
    if prot not in ["1e0m", "5wod", "1d5q"]:
        print("error at input prot ", prot)
        exit()


    weights_operator = get_weights_operator(input_operator)
       
    reader, scorefxn, straightPose, native = init_basics(prot)
   
    pdb_files = get_pdb_files(prot, input_source)
    print("run with ", len(pdb_files))
    
    # ITERATE THROUGHT INPUT PDBS
    results = []
    for file_id in pdb_files:
        try:
            model = reader.get_from_file(file_id)
        except:
            print("problem with ", file_id)
            continue
        init_score = scorefxn.score(model)
        init_rmsd = CA_rmsd(native, model)
        cost_func = NeuralNetworkFunction([model], scorefxn, native)
        score, rmsd = cost_func.score_and_rmsd(weights_operator)
        results.append({"id": file_id, "init_score": init_score,
                    "init_rmsd":init_rmsd, "score":score, "rmsd":rmsd})
    
    # init_score = scorefxn.score(straightPose)
    # init_rmsd = CA_rmsd(native, straightPose)
    # cost_func = NeuralNetworkFunction([straightPose], scorefxn, native)
    # score, rmsd = cost_func.score_and_rmsd(weights_operator)
    # results.append({"id": "straight", "init_score": init_score,
    #                 "init_rmsd":init_rmsd, "score":score, "rmsd":rmsd})

    # RESULTS 
    if "2Fases" in input_source:
        rosetta_structures = "Rosetta2Fases"
    else:
        rosetta_structures = "Rosetta4Fases"
        
    name = "results_folding_{}_{}_{}.csv".format(rosetta_structures, experiment, prot)
    print_results(name, results)
    

# def read_options_file(path):
#     config = configparser.ConfigParser()
#     config.read_file(open(path, "r")) 
#     operator = config.get('info', 'operator').strip()
#     prot = config.get('info', 'prot').strip()
#     experiment = config.get('info', 'experiment').strip()
#     source_pdb = config.get('info', 'source_pdb').strip()
#     return operator, prot, experiment, source_pdb
    
def main():
    #os.environ["job_id"] = str(random.randint(0,1000))
    os.environ["job_id"] = "1"
    os.environ["output_folder"] = "./" 

    # if input_with_options:
    input_operator = sys.argv[-4]
    prot = sys.argv[-3]
    experiment = sys.argv[-2]
    source_pdb = sys.argv[-1]
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

    run(input_operator, prot, experiment, source_pdb)
    
if __name__== "__main__":
    main()

