import numpy as np
from neuralnet import *

import os
import pyrosetta
import pyrosetta.rosetta as rosetta


from pyrosetta import (
    init, pose_from_sequence, pose_from_file, Pose, MoveMap, create_score_function, get_fa_scorefxn,
    MonteCarlo, TrialMover, SwitchResidueTypeSetMover, PyJobDistributor,
)

from pyrosetta.rosetta import core, protocols

class GreedyOperator():
    def __init__(self, weights = []):
        self.input_neurons = 4
        self.hidden_neurons = 2 
        self.steps = 10
        self.output_perturbation = 1
        self.disturb = [-0.01, 0.01, -0.05, 0.05, -0.25, 0.25, -1, 1,
                        -5, 5, -10, 10, -20, 20]
        self.neurons = [self.input_neurons, self.hidden_neurons, 1]
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

        for i in range(0, 1):
            chunks.append(weights[:self.hidden_neurons])
            weights = weights[self.hidden_neurons:]
    
        total_weights = chunks
        return(total_weights)
    

    def set_weights(self, weights_list):
        weights = self.__splitWeights(weights_list)
        #print(weights)
        self.nn = NeuralNetwork(self.neurons, weights)
        
    def init_random_weights(self):
        weights_num = (self.input_neurons*self.hidden_neurons)+(self.hidden_neurons*1)
        weights = []
        for i in list(range(0, weights_num)):
            weights.append( -2 * np.random.random_sample() + 1 )
 
        return self.__splitWeights(weights)

    def nn_connections(self):
        return (self.input_neurons*self.hidden_neurons) + (self.hidden_neurons*1)
 

    def __normalizeInputs(self, a):
        a = np.array(a , dtype="float64")
        a = np.interp(a, ( min(a), max(a) ), (-1, +1)  )
        a = a.tolist()
        # amin, amax = min(a), max(a)
        # if (amin != amax):
        #     for i, val in enumerate(a):
        #         a[i] = (val-amin) / (amax-amin)
        return a 
 
    # def __normalizeInputs(self, a):
    #     lower = -1
    #     upper = 1
    #     l_norm = [lower + (upper - lower) * x for x in a]
    #     # amin, amax = min(a), max(a)
    #     # if (amin != amax):
    #     #     for i, val in enumerate(a):
    #     #         a[i] = (val-amin) / (amax-amin)
    #     return l_norm
        

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
            model.assign(pose)
        
        #print(inputs)
        #return self.__normalizeInputs(inputs)
        #m_inputs = [round(i * 1000, 2) for i in inputs]
       
        return inputs
                
    def set_output(self, pose, dof, res, perturbation):
        if (dof == "phi"):
            pose.set_phi(res, pose.phi(res) + perturbation)
        if (dof == "psi"):
            pose.set_psi(res, pose.psi(res) + perturbation)


    def greedy_eval(self, inputs):
        p = inputs.index(min(inputs))
        return [self.disturb[p]]
        
    def render(self, pose, scorefxn, print_output = True):
        final_pose = Pose()
        final_pose.assign(pose)
        dofs = ["phi", "psi"]
        job_id = os.environ.get('job_id')
        dict_selected = {}
        for d in self.disturb:
            dict_selected[str(d)] = 0
            
        if (print_output):
            file_object = open("performance_ann_"+str(job_id)+".txt", "w") 
            
        for i in range(0, self.steps):
            for res in list(range(1, final_pose.total_residue() + 1)):
                for dof in dofs:
                    inputs = self.get_inputs(final_pose, scorefxn, dof, res )
                    if (print_output):
                        file_object.write("inputs res " + str(res) +" - " + str(dof) + " ")
                        file_object.write("[ ")
                        for val in inputs:
                            file_object.write(str(val))
                            file_object.write(" , ")
                        file_object.write("] \n")

                    if any(i < 0 for i in inputs):
                        output = self.greedy_eval(inputs)
                        dict_selected[str(output[0])] += 1
                        norm_output = self.__convert_output(output[0])
                        if (print_output):
                            file_object.write("**output **" + str(output) + "\n")
                        #print("**output **" , output)
                        self.set_output(final_pose, dof, res, norm_output)
                        #print(scorefxn.score(final_pose))
                        
        if (print_output):
            file_object.close()
            print("*** selected options *** ")
            print(dict_selected)
        return final_pose, scorefxn.score(final_pose)
                

    def __convert_output(self, OldValue):
       return OldValue
    
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
                        output = self.greedy_eval(inputs)
                        norm_output = self.__convert_output(output[0])
                        #print("**output **" , norm_output * self.output_perturbation)
                        self.set_output(model, dof, res, norm_output)

        return scorefxn.score(model)
                
