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
from get_models_from_rosetta import get_models_from_source


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
    popsize = 96                   # Population size, must be >= 4
    mutate = 0.9                 # Mutation factor [0,2]
    recombination = 0.9          # Recombination rate [0,1]
    maxiter = 100                # Max number of generations (maxiter)


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
                   native_path = None, exp = "2Fases"):

    dst_1e0m = get_models_from_source(exp, "1e0m")
    dst_5wod = get_models_from_source(exp, "5wod")
    dst_1d5q = get_models_from_source(exp, "1d5q")
 

    #dict_1e0m = glob.glob("./data_pdbs/folded_rosetta_1e0m/*.pdb")
    #dict_1e0m = glob.glob("./data_pdbs/resultados2FasesRosetta_1e0m/modelos_2Fases_1e0m/*.pdb")
 
    dict_1e0m = glob.glob(dst_1e0m + "/*.pdb")
    #dict_1e0m = ["./data_pdbs/folded_rosetta_1e0m/result_RMSD_pose2.060133.pdb", "data_pdbs/folded_rosetta_1e0m/result_RMSD_pose4.050978.pdb", "./data_pdbs/folded_rosetta_1e0m/result_RMSD_pose3.205730.pdb"]
    #dict_5wod = glob.glob("./data_pdbs/folded_rosetta_5wod/*.pdb")
    dict_5wod = glob.glob(dst_5wod + "/*.pdb")
    dict_1d5q = glob.glob(dst_1d5q + "/*.pdb")
 
    # dict_5wod = glob.glob("./data_pdbs/resultados2fasesrosetta_5wod/modelos_2fases_5wod/*.pdb")
 
    # dict_5wod = ["./data_pdbs/folded_rosetta_5wod/result_RMSD_pose1.292267.pdb", "data_pdbs/folded_rosetta_5wod/result_RMSD_pose2.287725.pdb", "./data_pdbs/folded_rosetta_5wod/result_RMSD_pose1.663407.pdb"]
    reader = StructureReader()
    model_pdbs = []
    if mode == "group":
        if prot == "1e0m":
            for model in dict_1e0m:
                pose = reader.get_from_file(model)
                model_pdbs.append(pose)

        if prot == "1d5q":
            for model in dict_1d5q:
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
    print("read pdb file ", sys.argv[-2])
    pdb_filename = sys.argv[-2]
    exec_mode = sys.argv[-1]
    
    output_folder = pdb_filename.split("/")[-1].split(".")[0]
    output_folder = exec_mode + "_" + output_folder
    os.makedirs(output_folder, exist_ok=True)
    os.environ["output_folder"] = output_folder 
    #job_id = os.environ.get('job_id')
    #if (job_id == None):
    if rank == 0:
        found = False
        while found != True:
            rnd_number = np.random.randint(1000)
            my_file = Path(output_folder + "/evolution_"+str(rnd_number)+".txt")
            if my_file.is_file() == False:
                found = True
        os.environ["job_id"] = str(rnd_number)
 
    # if (len(sys.argv) > 1):
    #     job_id = sys.argv[2]
    #     os.environ["job_id"] = job_id
    # else:
    #     os.environ["job_id"] = 0

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
    scorefxn.score(straightPose)


    # test ANNOperator

    # greedy_operator = GreedyOperator()
    
    # final_pose, score = greedy_operator.render(straightPose, scorefxn)

    # #DSSP = protocols.moves.DsspMover()
    # #DSSP.apply(final_pose)    # populates the final_pose's Final_Pose.secstruct
    # ss = final_pose.secstruct()
    # print("final SS ", ss)
    # rmsd = CA_rmsd(pose, final_pose)
    # print("final score %f - %f" % (score, rmsd) )
    # final_pose.dump_pdb("greedy_pose.pdb")
    # exit()
    # weights = [1 for i in range(0, (4*2) + (2*1) )]
    # ann_operator = ANNOperator()
    # ann_operator.set_weights(weights)
    # print("score :", ann_operator.apply(straightPose, scorefxn))
    
    # ann_operator2 = ANNOperator()
    # n_weights = read_evo_weights("./EjecucionesOldANN/10Pertur500gensArctan/ann_job_507.txt")
    # ann_operator2.set_weights(n_weights)
    # #print("score : ", ann_operator2.apply(straightPose, scorefxn))
    # final_pose, score = ann_operator2.render(straightPose, scorefxn)
    # print("score : ", score) 
    # final_pose.dump_pdb("minimized_start_native.pdb")
    
    # exit()

    exp = "4Fases"
    
    #prot = "1e0m" if "1e0m" in output_folder else "5wod"
    if "1e0m" in output_folder:
        prot = "1e0m"
    if "5wod" in output_folder:
        prot = "5wod"
    if "1d5q" in output_folder:
        prot = "1d5q"
        
    if exec_mode == "unfolded":
        mode = "single"
        with_straight = True
        native_path = pdb_filename
        model_pdbs = read_pdb_files(prot, mode, with_straight, native_path, exp)
        cost_func = NeuralNetworkFunction(model_pdbs, scorefxn, pose)
    if exec_mode == "3models":
        mode = "group"
        with_straight = False
        native_path = pdb_filename
        model_pdbs = read_pdb_files(prot, mode, with_straight, native_path, exp)
        cost_func = NeuralNetworkFunction(model_pdbs, scorefxn, pose)
    if exec_mode == "3modelsAndStraight":
        mode = "group"
        with_straight = True
        native_path = pdb_filename
        model_pdbs = read_pdb_files(prot, mode, with_straight, native_path, exp)
        cost_func = NeuralNetworkFunction(model_pdbs, scorefxn, pose)




    # cost_func = NeuralNetworkFunction([straightPose], scorefxn, pose)
        
    if rank == 0:
        start = time.time()
        configure_differential_evolution(pose, straightPose, cost_func, size)
        MasterProcess(size, cost_func).terminate()

        end = time.time()
        print("TOTAL TIME : ", str(end - start))
    else:
        worker = Worker(rank, size, cost_func)
        worker.run()

if __name__== "__main__":
    main()

#print("acaba")

#rosetta.init()
#mnist = tf.keras.datasets.mnist
