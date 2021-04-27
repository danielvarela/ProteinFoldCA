#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7


from __future__ import print_function

# example to run the script: 
# python perform_folding_at_pdbs.py 1e0m unfolded

#import pyrosetta as rosetta
#import tensorflow as tf

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
    "5wod": "./data_pdbs/5WOD/5wod_A.pdb"
}

rosetta_output = {
    "1e0m": "./Rosetta2Fases_PDBS_1e0m/",
    "5wod": "./Rosetta2Fases_PDBS_5wod/"
}


# rosetta_output = {
#     "1e0m": "./resultadosBuenosRosetta_1e0m/",
#     "5wod": "./resultadosBuenosRosetta_5wod/"
# }


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


def get_job(prot, experiment, list_bests):
    job_list = ["unfolded", "3models", "3modelsAndStraight"]
    with open(list_bests, "r") as flist:
        bests = [l.strip() for l in flist.readlines()]



    by_prot = [e for e in bests if prot in e]
    by_experiment = ""
    for p in by_prot:
        for opt in p.split("/"):
            if ("_") in opt:
                words = opt.split("_")
                if experiment in words:
                    by_experiment = p
                    break
                
    # by_experiment = [e for e in by_prot if experiment in [d.split("_") for d in e.split("/") if any(ext in d for ext in job_list)]]
    if len(by_experiment) == 0:
        print("job not found")
        exit()
    else:
        selected_job = by_experiment
        selected_job = selected_job.replace("evolution", "ann_job")
        if not os.path.isfile(selected_job):
            print("selected job {} not exists ".format(selected_job))
            exit()
    return selected_job
    
        
def main():
    #os.environ["job_id"] = str(random.randint(0,1000))
    os.environ["job_id"] = "1"
    os.environ["output_folder"] = "./" 

    list_bests = sys.argv[-3]
    prot = sys.argv[-2]
    experiment = sys.argv[-1]

    if prot not in ["1e0m", "5wod"]:
        print("error at input prot ", prot)
        exit()

    lista = list_bests.split("_")[-1].split(".")[0]
    results_csv = open("results_folding_"+lista+"_"+experiment+"_"+prot+".csv", "w")

if __name__== "__main__":
    main()

