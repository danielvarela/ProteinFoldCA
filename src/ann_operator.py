import numpy as np
from neuralnet import *

import os
import pyrosetta
import pyrosetta.rosetta as rosetta
import math

from pyrosetta import (
    init, pose_from_sequence, pose_from_file, Pose, MoveMap, create_score_function, get_fa_scorefxn,
    MonteCarlo, TrialMover, SwitchResidueTypeSetMover, PyJobDistributor,
)

from pyrosetta.rosetta import core, protocols

ref_wts =  {"env" : 1.0, "pair" : 1.0, "cbeta":1.0, "vdw":1.0, "rg":3.0, "cenpack" :1.0, "hs_pair":1.0, "ss_pair":1.0, "rsigma" : 1.0, "sheet" : 1.0} 

class ANNOperator():
    def __init__(self, weights = []):
        self.disturb = [-0.05, 0.05, -1, 1, -5, 5, -10, 10]
        #self.disturb = [-0.05, 0.05, -5, 5, -10, 10, -20, 20]
        self.input_neurons = len(self.disturb)
        self.hidden_neurons = 10
        #self.output_neurons = self.input_neurons
        self.output_neurons = 1
        self.steps = 20
        self.output_perturbation = 10
        self.output_folder = os.environ.get('output_folder')

        self.neurons = [self.input_neurons, self.hidden_neurons, self.output_neurons]
        if (len(weights) < 1):
            self.init_weights = self.init_random_weights()
        else:
            self.init_weights = weights
            
        self.nn = NeuralNetwork(self.neurons, self.init_weights)

    def __splitWeights(self, weights):
        chunks = []
        for i in range(0, self.hidden_neurons):
            chunks.append(weights[:self.input_neurons])
            weights = weights[self.input_neurons:]

        for i in range(0, self.output_neurons):
            chunks.append(weights[:self.hidden_neurons])
            weights = weights[self.hidden_neurons:]
    
        total_weights = chunks
        return(total_weights)
    

    def set_weights(self, weights_list):
        weights = self.__splitWeights(weights_list)
        #print(weights)
        self.nn = NeuralNetwork(self.neurons, weights)
        
    def init_random_weights(self):
        weights_num = self.nn_connections()
        weights = []
        for i in list(range(0, weights_num)):
            weights.append( -2 * np.random.random_sample() + 1 )
 
        return self.__splitWeights(weights)

    def nn_connections(self):
        return (self.input_neurons*self.hidden_neurons) + (self.hidden_neurons*self.output_neurons)
    

    def __normalizeInputs(self, a):
        a = np.array(a , dtype="float64")
        a = np.interp(a, ( min(a), max(a) ), (-1, +1)  )
        a = a.tolist()
        # amin, amax = min(a), max(a)
        # if (amin != amax):
        #     for i, val in enumerate(a):
        #         a[i] = (val-amin) / (amax-amin)
        return a 
        
    def get_greedy_inputs(self, pose, scorefxn, dof, res):
        model = Pose()
        model.assign(pose)
        disturb = self.disturb[:2] 
        start_score = scorefxn.score(model)
        final_inputs = []
        for p in disturb:
            if (dof == "phi"):
                model.set_phi(res, model.phi(res) + p)
                #print("current_dof ", model.phi(res))
            if (dof == "psi"):
                model.set_psi(res, model.psi(res) + p)
                #print("current_dof ", model.phi(res))

            greedy_pose = Pose()
            greedy_pose.assign(model)
            inputs = []
            inputs_score = []
            for perturbation in disturb:
                if (dof == "phi"):
                    greedy_pose.set_phi(res + 1, greedy_pose.phi(res + 1) + perturbation)
                    #print("greedy_dof ", greedy_pose.phi(res + 1))
                if (dof == "psi"):
                    greedy_pose.set_psi(res + 1, greedy_pose.psi(res + 1) + perturbation)
                    #print("greedy_dof ", greedy_pose.psi(res + 1))
                    
                score = scorefxn.score(greedy_pose)
                inputs.append(score)
                inputs_score.append(score - start_score)
                #print(score)
                greedy_pose.assign(model)

            model.assign(pose)

            min_prev = inputs.index(min(inputs)) 
            final_inputs.append(inputs_score[min_prev])

        return final_inputs
 
    def get_inputs(self, pose, scorefxn, dof, res):
        model = Pose()
        model.assign(pose)

        init_score = scorefxn.score(model, res)
        #print("init score ", init_score)
        #disturb = [-2, 2, -4, 4]
        disturb = self.disturb
        inputs = []
        for perturbation in disturb:
            if (dof == "phi"):
                model.set_phi(res, model.phi(res) + perturbation)
            if (dof == "psi"):
                model.set_psi(res, model.psi(res) + perturbation)

            score = scorefxn.score(model, res)

            inputs.append(score - init_score)
            #model.dump_pdb("step50_at_" + str(dof) + "_" + str(res) + ".pdb" )
            model.assign(pose)
        
        #if (res < model.total_residue()):
        #    greedy_inputs = self.get_greedy_inputs(model, scorefxn, dof, res)
        #else:
        #    greedy_inputs = [0,0]

        #print(greedy_inputs)
        #inputs = inputs + greedy_inputs
        #print(inputs)

        return self.__normalizeInputs(inputs)
        #m_inputs = [round(i * 1000, 2) for i in inputs]
        #print(inputs)
        #return inputs
                
    def set_output(self, pose, dof, res, perturbation):
        if (dof == "phi"):
            pose.set_phi(res, pose.phi(res) + perturbation)
        if (dof == "psi"):
            pose.set_psi(res, pose.psi(res) + perturbation)


    def read_pose_scores(self, pose):
        pose_scores = []
        energies = pose.energies().total_energies()
        energies_as_list = [i.strip('( )') for i in str(energies).split(') (')]
        
        for e in energies_as_list:
            term, unweighted_val = e.split()
            term = term.replace(';', '')
            if term in ref_wts:
                weighted_val = float(unweighted_val) * ref_wts[term]
                pose_scores.append(': '.join([term,str(weighted_val)]))
                
        return pose_scores


    def render(self, pose, scorefxn):
        final_pose = Pose()
        final_pose.assign(pose)
        dofs = ["phi", "psi"]
        dict_selected = {}
        for d in self.disturb:
            dict_selected[str(d)] = 0
 
        min_inputs = {}
        for d in self.disturb:
            min_inputs[str(d)] = 0

        move_selected = []
        job_id = os.environ.get('job_id')
        file_object = open(self.output_folder + "/performance_ann_"+str(job_id)+".txt", "w") 
        for i in range(0, self.steps):
            for res in list(range(1, final_pose.total_residue() + 1)):
                for dof in dofs:
                    inputs = self.get_inputs(final_pose, scorefxn, dof, res )
                    min_inputs[str(self.disturb[inputs.index(min(inputs))])] += 1
                    if any(i < 0 for i in inputs):
                        output = self.nn.eval(inputs)
                        norm_output = self.__convert_output(output)
                        file_object.write("**output **" + str(norm_output) + "\n")
                        if self.output_neurons > 1:
                            dict_selected[str(norm_output)] += 1
                        else:
                            move_selected.append(norm_output)
                              
                        #print("**output **" , output)
                        self.set_output(final_pose, dof, res, norm_output)

                file_object.write("*** at res " + str(res) +  " *** \n")
                scorefxn.score(final_pose)
                w_scores = self.read_pose_scores(final_pose)
                file_object.write(str(w_scores))
                file_object.write(" \n")
            final_pose.dump_pdb("final_pose_step"+str(i)+".pdb")
                        
        file_object.write("*** min inputs *** \n")
        file_object.write(str(min_inputs))
 
        file_object.write(" \n")

        if self.output_neurons > 1:
            file_object.write("*** selected options *** \n")
            file_object.write(str(dict_selected))
            file_object.write(" \n")
        else:
            for j in move_selected:
                file_object.write(str(j))
                file_object.write(",")
            file_object.write("\n")

        file_object.close()
        return final_pose, scorefxn.score(final_pose)
                

    def __convert_output(self, OldValue):
        NewValue = 0
        if (self.output_neurons == 1):
            OldValue = OldValue[0]
            OldMin = (math.pi / 2) * -1
            OldMax = (math.pi / 2) 
            NewMin = -1 * self.output_perturbation
            NewMax = 1 * self.output_perturbation
            NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax -
                                                                    OldMin)) + NewMin
        else:
            p = OldValue.index(min(OldValue))
            NewValue = self.disturb[p]
 
        return NewValue
    
    def apply(self, pose, scorefxn):
        model = Pose()
        model.assign(pose)
        dofs = ["phi", "psi"]
        for i in range(0, self.steps):
            for res in list(range(1, model.total_residue() + 1)):
                for dof in dofs:
                    inputs = self.get_inputs(model, scorefxn, dof, res )
                    #print("inputs res %i - %s :" % (res, dof), end = " ")
                    #print(inputs)
                    #if all(i != 0 for i in inputs):
                    if any(i < 0 for i in inputs):
                        output = self.nn.eval(inputs)
                        norm_output = self.__convert_output(output)
                        #print("**output **" , norm_output * self.output_perturbation)
                        self.set_output(model, dof, res, norm_output)

        return scorefxn.score(model)
                
