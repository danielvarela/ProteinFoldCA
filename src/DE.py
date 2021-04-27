import random

#import gym
#from gym import spaces
import time
import numpy as np
from threading import Thread
#from visual_utils.bokeh_visualizer import Visualizer
#from cost_funcs.psp_func import PSPFunction
#from cost_funcs.nn_folding_func import NeuralNetworkFunction 

import os
from mpi_rosetta import *

def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector 
    for i,k in enumerate(vec):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
        
    return vec_new


class Individual():
    def __init__(self, genotype, score):
        self.genotype = genotype
        self.score = score
        
#--- MAIN ---------------------------------------------------------------------+

class DifferentialEvolutionAlgorithm():
    def __init__(self, cost_func, popsize, mutate, recombination, maxiter,
                 world_size = 1):
        #func = NeuralNetworkFunction(seq)
        #cost_func = func                   # Cost function
        self.cost_func = cost_func
        #self.cost_func = cost_func
        self.popsize = popsize
        self.mutate = mutate
        self.recombination = recombination
        self.maxiter = maxiter
        self.world_size = world_size
        self.job_id = os.environ.get('job_id')
        self.output_folder = os.environ.get('output_folder')
        if (self.job_id == None):
            self.job_id = str(0)
        
    def main(self):
        #--- INITIALIZE A POPULATION (step #1) ----------------+
        #self.cost_func = PSPFunction(self.seq)
        #print("START DE")
        ind_size = self.cost_func.size()
        self.bounds = [(-1,1)] * ind_size            # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
        population = []

        popul = []
        for i in range(0,self.popsize):
            indv = []
            for j in range(len(self.bounds)):
                indv.append(random.uniform(self.bounds[j][0],self.bounds[j][1]))
            popul.append(indv)
                
        popul_scores = MasterProcess(self.world_size, self.cost_func).run(popul)
 
        #print("POPUL SCORES : ", popul_scores) 
        for i in range(0, self.popsize):
            indv_score = popul_scores[i]
            build_ind = Individual(popul[i], indv_score)
            population.append(build_ind)
               
        #--- SOLVE --------------------------------------------+

        

        start = time.time()
        # cycle through each generation (step #2)
        for i in range(1,self.maxiter+1):
            #print('GENERATION:' + str(i))
            file_object = open(self.output_folder + "/evolution_" + str(self.job_id) + ".txt", "a")
            file_object.write('GENERATION:' + str(i) + '\n')

            gen_scores = [] # score keeping
            
            # cycle through each individual in the population
            trials = []
            for j in range(0, self.popsize):

                #--- MUTATION (step #3.A) ---------------------+

                # select three random vector index positions [0, self.popsize), not including current vector (j)
                candidates = list(range(0,self.popsize))
                candidates.remove(j)
                random_index = random.sample(candidates, 3)

                x_1 = population[random_index[0]].genotype
                x_2 = population[random_index[1]].genotype
                x_3 = population[random_index[2]].genotype
                x_t = population[j].genotype     # target individual

                # subtract x3 from x2, and create a new vector (x_diff)
                x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

                # multiply x_diff by the mutation factor (F) and add to x_1
                v_donor = [x_1_i + self.mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                v_donor = ensure_bounds(v_donor, self.bounds)

                #--- RECOMBINATION (step #3.B) ----------------+

                v_trial = []
                #print("len v_donor ", len(v_donor))
                for k,obj in enumerate(x_t):
                    crossover = random.random()
                    if crossover <= self.recombination:
                        v_trial.append(v_donor[k])

                    else:
                        v_trial.append(x_t[k])

                #print("trials " , j)
                trials.append(v_trial)
                    
                #--- GREEDY SELECTION (step #3.C) -------------+
            #print(len(trials))

            trial_scores = MasterProcess(self.world_size, self.cost_func).run(trials)
                
            for j in range(0, self.popsize):
                score_trial  = trial_scores[j] 
                score_target = population[j].score

                if score_trial < score_target:
                    population[j] = Individual(trials[j], score_trial)
                    gen_scores.append(score_trial)
                else:
                    gen_scores.append(score_target)

            
            #--- SCORE KEEPING --------------------------------+
            

            #print("POPUL SCORES : ", gen_scores) 
            gen_avg = sum(gen_scores) / self.popsize                         # current generation avg. fitness
            gen_best = min(gen_scores)                                  # fitness of best individual
            gen_sol = population[gen_scores.index(min(gen_scores))].genotype     # solution of best individual

            file_object.write('      > GENERATION AVERAGE: %f \n' % gen_avg )
            file_object.write('      > GENERATION BEST: %f \n' % gen_best )
            file_object.close()
            self.cost_func.render(i, gen_sol)        

        done = time.time()
        elapsed = done - start
        print("****** DE TIME ************* ")
        print(elapsed) 



        start = time.time()
        # self.cost_func.print_popul(population)
        scores, rmsds = MasterProcess(self.world_size, self.cost_func).final_popul(population)
        self.cost_func.print_calculated_popul(scores, rmsds)
        done = time.time()
        elapsed = done - start
        print("****** PRINT POPUL TIME ************* ")
        print(elapsed) 
        return gen_sol

#--- MAIN  ----------------------------------------------------------------+
    
def main():
    #seq = 'HHHHHPHHHHHHPHHHHPHH' # Our input sequence
    #seq = 'HPHPPHHPHPPHPHHPPHPH' # Our input sequence
    seq = 'HPHPHPHPPHPH' # Our input sequence
    #func = CostFunction()
    #ind_size = func.size()
    #print(ind_size)
    #bounds = [(-1,1)] * ind_size           # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
    #popsize = len(seq) * 15            # Population size, must be >= 4
    popsize = 100                   # Population size, must be >= 4
    mutate = 0.3                  # Mutation factor [0,2]
    recombination = 0.9          # Recombination rate [0,1]
    maxiter = 5000                        # Max number of generations (maxiter)

    strategy  = "nn_operator_ext"
    mode = "nn_folding"
    cost_func = 1
    
    #--- RUN ----------------------------------------------------------------------+
    alg = DifferentialEvolutionAlgorithm(seq, cost_func, popsize, mutate, recombination,
                                         maxiter)
    alg.main()
    #thread = Thread(target=alg.main)
    #thread.start()


if __name__== "__main__":
  main()


#--- END ----------------------------------------------------------------------+
