#!usr/bin/env python


import optparse    # for option sorting
import sys
import pyrosetta
import pyrosetta.rosetta as rosetta


from pyrosetta import (
    init, pose_from_sequence, pose_from_file, Pose, MoveMap, create_score_function, get_fa_scorefxn,
    MonteCarlo, TrialMover, SwitchResidueTypeSetMover, PyJobDistributor
)

from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.core.scoring import CA_rmsd

from ann_operator import ANNOperator
from greedy_operator import GreedyOperator
import numpy as np
from neuralnet import *
import os

class FragmentOperator(ANNOperator):
    def __init__(self, weights = []):
        super(FragmentOperator, self).__init__(weights)
        self.input_neurons = 200 
        self.hidden_neurons = 100
        #self.output_neurons = self.input_neurons
        self.output_neurons = 200
        self.neurons = [self.input_neurons, self.hidden_neurons, self.output_neurons]
        if (len(weights) < 1):
            self.init_weights = self.init_random_weights()
        else:
            self.init_weights = weights
            
        self.nn = NeuralNetwork(self.neurons, self.init_weights)
        self.greedy_operator = GreedyOperator()
 

        ## Preparing Fragment Mover
        # 1. MoveMap
        movemap = MoveMap()
        movemap.set_bb(True)
        
        # 2. Prepare Fragset
        long_frag_length = 9
        if "1e0m_A.pdb" in sys.argv[1]:
            long_frag_filename = "./data_pdbs/1E0M/aa1E0M_09_05.200_v1_3"
            short_frag_filename = "./data_pdbs/1E0M/aa1E0M_03_05.200_v1_3" 
        else:
            long_frag_filename = "./data_pdbs/5WOD/aa5WOD_09_05.200_v1_3"
            short_frag_filename = "./data_pdbs/5OD/aa5WOD_03_05.200_v1_3" 
            
        # print("size ", fragset_long.size())
        # print("nr_frames ", fragset_long.nr_frames())
        # print("min_pos ", fragset_long.min_pos())
        # print("max_pos ", fragset_long.max_pos())

        ## Preparing Fragment Mover

        # 1. MoveMap
        movemap = MoveMap()
        movemap.set_bb(True)
        
        # 2. Prepare Fragset
        # self.fragset_short = core.fragment.ConstantLengthFragSet( 3 , short_frag_filename )
        
        # self.fragset_long = core.fragment.ConstantLengthFragSet( 9 , long_frag_filename )
        
        # print("size ", fragset_long.size())
        # print("nr_frames ", fragset_long.nr_frames())
        # print("min_pos ", fragset_long.min_pos())
        # print("max_pos ", fragset_long.max_pos())
        
    def __normalizeInputs(self, a):
        a = np.array(a , dtype="float64")
        a = np.interp(a, ( min(a), max(a) ), (-1, +1)  )
        a = a.tolist()
        # amin, amax = min(a), max(a)
        # if (amin != amax):
        #     for i, val in enumerate(a):
        #         a[i] = (val-amin) / (amax-amin)
        return a 
     
    def run_frag(self, final_pose, scorefxn, len_frag = 9, render = False):
        # 3. Obtain Fragmelist for the current seqpos 
        if render:
            job_id = os.environ.get('job_id')
            file_object = open("performance_ann_"+str(job_id)+".txt", "w") 
        for start_pos in range(1, 9, 1):
            final_pose, tmp_score = self.greedy_operator.render(final_pose, scorefxn, False)
            for seqpos in range(start_pos, self.fragset_long.max_pos(), len_frag):
                framelist = core.fragment.FrameList()

                if (len_frag==9):
                    self.fragset_long.frames(seqpos,framelist)
                else:
                    self.fragset_short.frames(seqpos,framelist)
                    
                # 4. Iterate throught frags to obtain the energy differences
            
                current_model = Pose()
                current_model.assign(final_pose)
                #init_score = scorefxn.score(current_model)
                init_score = final_pose.energies().total_energy() 
                #print(init_score)
                inputs = []
                for frame in framelist:
                    for i in range(1, frame.nr_frags()+1):
                        model = Pose()
                        model.assign(current_model)
                        frame.apply(i, model)
                        score = scorefxn.score(model)
                        #print("i : %i , score %f " % (i, score))
                        inputs.append(score - init_score)
                        
                #print("inputs " , len(inputs))
                #print(inputs)
            
                # 5. Apply the best frame
                
                #print("seqpos ", seqpos, end=" : ")
                #print("init_score ", init_score, end=" -> ")
                #print("best operation ", min(inputs))
                # 5.1 greedy strategy
                if (len(inputs) > 0):
                    inputs = self.__normalizeInputs(inputs)
                    output = self.nn.eval(inputs)
                    # 5.1 greedy strategy
                    #best_frame = inputs.index(min(inputs)) + 1
                    # 5.2 nn strategy
                    best_frame = output.index(min(output)) + 1

                    for frame in framelist:
                        frame.apply(best_frame, current_model)
                        score = scorefxn.score(current_model)
                        #print("final score  ", score)

                    if render:
                        w_scores = self.read_pose_scores(current_model)
                        file_object.write(str(w_scores))
                        file_object.write(" \n")
                    final_pose.assign(current_model)
                
        if render:
            file_object.close()
        return final_pose

    # def read_pose_scores(self, pose):
    #      pose_scores = []
    #      energies = pose.energies().total_energies()
    #      energies_as_list = [i.strip('( )') for i in str(energies).split(') (')]
        
    #      for e in energies_as_list:
    #          term, unweighted_val = e.split()
    #          term = term.replace(';', '')
    #          if term in ref_wts:
    #              weighted_val = float(unweighted_val) * ref_wts[term]
    #              pose_scores.append(': '.join([term,str(weighted_val)]))

    #     return pose_scores

        
    def render(self, pose, scorefxn, render = True):
        final_pose = Pose()
        final_pose.assign(pose)
        final_pose = self.run_frag(final_pose, scorefxn, 9, render)
        #final_pose = self.run_frag(final_pose, scorefxn, 3, render)
        return final_pose, scorefxn.score(final_pose)
 
    def apply(self, pose, scorefxn):
        final_pose, score = self.render(pose, scorefxn, False)
        return score
    
