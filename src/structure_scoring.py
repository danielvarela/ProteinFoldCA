#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import os
import time
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

from ann_operator import ANNOperator
from FragmentLibraryMover import FragmentOperator
from greedy_operator import GreedyOperator



class PoseScoring():
    def __init__(self, native):
        self.scorefxn = create_score_function('score3')
        self.native = native
        self.DSSP = protocols.moves.DsspMover()
        self.DSSP.apply(native)    # populates the native's Native.secstruct
        self.native_ss = native.secstruct()
 
    def score(self, pose, current_res = -1):
        for i in range(0, len(self.native_ss)):
            pose.set_secstruct(i + 1, self.native_ss[i])
        
        pose_score = self.scorefxn(pose)

        # if (current_res > 0):
        #     if (current_res < 6):
        #         init = 0
        #     else:
        #         init = current_res - 5
                
        #     pose_score = CA_rmsd(self.native, pose, init, current_res + 5)
        # else:
        #     pose_score = CA_rmsd(self.native, pose)
        
        #self.DSSP.apply(pose)    # populates the native's Native.secstruct
        #ss = pose.secstruct()
        
        #penalize = 0
        #for i in range(0, len(ss)):
            #if (ss[i] != self.native_ss[i]):
            #    penalize += 100
 
        #pose_score = pose_score + penalize
        
        # #obtain the pose Energies object and all the residue total scores
        # energies = pose.energies()
        # residue_energies = [energies.residue_total_energy(i)
        #                     for i in range(1, pose.total_residue() + 1)]


        # #obtain the non-zero weights of the ScoreFunction, active ScoreTypes
        # weights = [core.scoring.ScoreType(s)
        # for s in range(1, int(core.scoring.end_of_score_type_enumeration) + 1)
        #            if self.scorefxn.weights()[core.scoring.ScoreType(s)]]

        # residue_weighted_energies_matrix = [
        #     [energies.residue_total_energies(i)[w] * self.scorefxn.weights()[w]
        #      for i in range(1, pose.total_residue() + 1)]
        #     for w in weights]

        #self.display_score(pose_score, residue_energies, weights, residue_weighted_energies_matrix)
        #self.display_score(pose_score)
        return pose_score
 
    def display_score(self, total_score):
        print("*** TOTAL SCORE: " + str(total_score) + "***")
        
    def display_score_info(self, total_score, residue_energies, weights, residue_weighted_energies_matrix):
        display_residues = list(range(1, pose.total_residue() + 1))
        # output information on the requested residues
        for i in display_residues:
            print( '='*80 )
            print( 'Pose numbered Residue' , i )
            print( 'Total Residue Score:' , residue_energies[i-1] )
            print( 'Score Breakdown:\n' + '-'*45 )
            # loop over the weights, extract the scores from the matrix
            for w in range(len(weights)):
                print( '\t' + core.scoring.name_from_score_type(weights[w]).ljust(20) + ':\t' ,\
                       residue_weighted_energies_matrix[w][i-1] )
            print( '-'*45 )
        print("*** TOTAL SCORE: " + str(total_score) + "***")
    print( '='*80 )



class NeuralNetworkFunction():
    def __init__(self, poses, scorefxn, native):
        self.ann_operator = ANNOperator()
        # self.ann_operator = FragmentOperator()
        self.poses = []
        for pose in poses:
            model = Pose()
            model.assign(pose)
            self.poses.append(model)
        self.native = Pose()
        self.native.assign(native)
        self.scorefxn = scorefxn
        self.job_id = os.environ.get('job_id')
        self.output_folder = os.environ.get('output_folder')
        if (self.job_id == None):
            self.job_id = 0
        
    def size(self):
        return self.ann_operator.nn_connections()
    
    def score_and_rmsd(self, x):
        #print("run " + str(len(x)) )
        self.ann_operator.set_weights(x)
        results = []
        for pose in self.poses:
            model = Pose()
            model.assign(pose)
            
            # reward is energy value of the folded protein
            final_pose, reward = self.ann_operator.render(model, self.scorefxn)
            rmsd = CA_rmsd(self.native, final_pose)
            results.append({"pose":final_pose, "reward" : reward, "rmsd":rmsd})
            
        #reward = sum([r["reward"] for r in results])
        reward = min([r["reward"] for r in results])
        rmsd = min([r["rmsd"] for r in results])
        return reward, rmsd


    def run(self, x):
        #print("run " + str(len(x)) )
        self.ann_operator.set_weights(x)
        
        reward = 0
        for pose in self.poses:
            model = Pose()
            model.assign(pose)

            # reward is energy value of the folded protein
            reward += self.ann_operator.apply(model, self.scorefxn)
        return reward

    def print_calculated_popul(self, scores, rmsds):
        with open(self.output_folder + "/evolution_"+str(self.job_id)+".txt", "a") as myfile:
            myfile.write("SCORES: " )
            for i in range(0, len(scores)):
                myfile.write(str(scores[i]) + " , " )
            myfile.write("\n")
            myfile.write("RMSDS: " )
            for i in range(0, len(rmsds)):
                myfile.write(str(rmsds[i]) + " , " )
            myfile.write("\n")


    def print_popul(self, popul):
        scores = []
        rmsds = []
        for x in popul:
            self.ann_operator.set_weights(x.genotype)
            results = []
            for pose in self.poses:
                model = Pose()
                model.assign(pose)

                # reward is energy value of the folded protein
                final_pose, reward = self.ann_operator.render(model, self.scorefxn)
                rmsd = CA_rmsd(self.native, final_pose)
                results.append({"pose":final_pose, "reward" : reward, "rmsd":rmsd})

            #reward = sum([r["reward"] for r in results])
            reward = min([r["reward"] for r in results])
            rmsd = min([r["rmsd"] for r in results])
            scores.append(reward)
            rmsds.append(rmsd)

        with open(self.output_folder + "/evolution_"+str(self.job_id)+".txt", "a") as myfile:
            myfile.write("SCORES: " )
            for i in range(0, len(scores)):
                myfile.write(str(scores[i]) + " , " )
            myfile.write("\n")
            myfile.write("RMSDS: " )
            for i in range(0, len(rmsds)):
                myfile.write(str(rmsds[i]) + " , " )
            myfile.write("\n")

            
    def render(self, gen, x): 
        start = time.time()
        self.ann_operator.set_weights(x)
        
        results = []
        for pose in self.poses:
            model = Pose()
            model.assign(pose)

            # reward is energy value of the folded protein
            final_pose, reward = self.ann_operator.render(model, self.scorefxn)
            rmsd = CA_rmsd(self.native, final_pose)
            results.append({"pose":final_pose, "reward" : reward, "rmsd":rmsd})
            
        
        
        reward = sum([r["reward"] for r in results])
        rmsd = min([r["rmsd"] for r in results])
        final_pose = [r["pose"] for r in results if r["rmsd"] == rmsd][0]
        #DSSP = protocols.moves.DsspMover()
        #DSSP.apply(final_pose)    # populates the native's Native.secstruct
        ss = final_pose.secstruct()
 
        with open(self.output_folder + "/evolution_"+str(self.job_id)+".txt", "a") as myfile:
            myfile.write("BEST IND : " + str(self.scorefxn.score(final_pose)) +
            " " + str(rmsd) + " ss " + ss + "\n")
        file_object = open(self.output_folder + "/ann_job_" + str(self.job_id) + ".txt", "w")
        file_object.write( str("gen: "+ str(gen)) + "\n" )
        for i in x:
            file_object.write(str(i) + "\n")
        
        file_object.close()
        final_pose.dump_pdb(self.output_folder + '/minimized' + str(self.job_id) + '.pdb')
        done = time.time()
        elapsed = done - start
        print("****** INDIVIDUAL TIME ************* ")
        print(elapsed) 
