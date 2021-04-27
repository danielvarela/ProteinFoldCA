#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7


from __future__ import print_function

#import pyrosetta as rosetta
#import tensorflow as tf

import glob
import optparse    # for option sorting
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

import time
from DE import *


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
 

#init(extra_options = "-constant_seed")  # WARNING: option '-constant_seed' is for testing only! MAKE SURE TO REMOVE IT IN PRODUCTION RUNS!!!!!
init(extra_options = "-mute all")  # WARNING: option '-constant_seed' is for testing only! MAKE SURE TO REMOVE IT IN PRODUCTION RUNS!!!!!
import os;
#os.chdir('.test.output')

from structure_reader import StructureReader, StraightMover
from structure_scoring import PoseScoring, NeuralNetworkFunction
        
def configure_differential_evolution(native, straightPose, cost_func, world_size):
    popsize = 1                   # Population size, must be >= 4
    mutate = 0.9                 # Mutation factor [0,2]
    recombination = 0.9          # Recombination rate [0,1]
    maxiter = 2                # Max number of generations (maxiter)


    #--- RUN ------------------------------------------------------+
    alg = DifferentialEvolutionAlgorithm(cost_func, popsize, mutate, recombination,
                                             maxiter, world_size)
    alg.main()



def read_evo_weights(file_weights):
    with open(file_weights, "r") as file_object:
        weights = [w.strip() for w in file_object.readlines()[1:]]
        n_weights = np.array(weights, dtype="float64")

    return n_weights
    
def read_pdb_files(prot = "1e0m", mode ="group", with_straight = False,
                   native_path = None):

    #dict_1e0m = glob.glob("./data_pdbs/folded_rosetta_1e0m/*.pdb")
    dict_1e0m = glob.glob("./data_pdbs/resultados2FasesRosetta_1e0m/modelos_2Fases_1e0m/*.pdb")
    #dict_1e0m = ["./data_pdbs/folded_rosetta_1e0m/result_RMSD_pose2.060133.pdb", "data_pdbs/folded_rosetta_1e0m/result_RMSD_pose4.050978.pdb", "./data_pdbs/folded_rosetta_1e0m/result_RMSD_pose3.205730.pdb"]
    #dict_5wod = glob.glob("./data_pdbs/folded_rosetta_5wod/*.pdb")
    dict_5wod = glob.glob("./data_pdbs/resultados2FasesRosetta_5wod/modelos_2Fases_5wod/*.pdb")

    # dict_5wod = ["./data_pdbs/folded_rosetta_5wod/result_RMSD_pose1.292267.pdb", "data_pdbs/folded_rosetta_5wod/result_RMSD_pose2.287725.pdb", "./data_pdbs/folded_rosetta_5wod/result_RMSD_pose1.663407.pdb"]
    reader = StructureReader()
    model_pdbs = []
    if mode == "group":
        if prot == "1e0m":
            for model in dict_1e0m:
                pose = reader.get_from_file(model)
                model_pdbs.append(pose)

        if prot == "5wod":
            for model in dict_5wod:
                pose = reader.get_from_file(model)
                model_pdbs.append(pose)

    if with_straight:
        pose = reader.get_from_file(native_path)
        straightMover = StraightMover(pose)
        straightPose = straightMover.get_model()
        model_pdbs.append(straightPose)
 

    return model_pdbs

def main():
    # PDB file option   
    #print("RANK ", rank)
    ref_wts =  {"env" : 1.0, "pair" : 1.0, "cbeta":1.0, "vdw":1.0, "rg":3.0, "cenpack" :1.0, "hs_pair":1.0, "ss_pair":1.0, "rsigma" : 1.0, "sheet" : 1.0} 


    print("read pdb file ", sys.argv[-1])
    pdb_filename = sys.argv[-1]

    # create a pose from the desired PDB file
    # load the data from pdb_file into the pose
    reader = StructureReader()
    pose = reader.get_from_file(pdb_filename)
    
    #reader.pose_structure(pose)
    straightMover = StraightMover(pose)
    straightPose = straightMover.get_model()
    #reader.pose_structure(straightPose)
        
    #reader.pose_structure(pose)
    scorefxn = PoseScoring(pose)
    scorefxn.score(pose)

    pose_scores = []
    energies = pose.energies().total_energies()
    energies_as_list = [i.strip('( )') for i in str(energies).split(') (')]
            
    for e in energies_as_list:
        term, unweighted_val = e.split()
        term = term.replace(';', '')
        if term in ref_wts:
            weighted_val = float(unweighted_val) * ref_wts[term]
            #pose_scores.append(': '.join([term,str(weighted_val)]))
            pose_scores.append(weighted_val)

    print(pose_scores)
    exit()

if __name__== "__main__":
    main()

#print("acaba")

#rosetta.init()
#mnist = tf.keras.datasets.mnist
