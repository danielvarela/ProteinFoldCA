import time
from mpi4py import MPI
import numpy as np
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
 
class MasterProcess():
    def __init__(self, size, cost_func):
        self.rank = 0
        self.size = size
        self.cost_func = cost_func
        
    def __make_chunks(self, popul, size):
        lst = copy.deepcopy(popul)
        for d in range(0, len(lst)):
            lst[d] = [d] + lst[d]
        size = size
        chunks = [lst[i:i + size] for i in range(0, len(lst), size)]
        return chunks
                    
    def __make_chunks_popul(self, individual_population, size):
        popul = [x.genotype for x in individual_population]
        lst = copy.deepcopy(popul)
        for d in range(0, len(lst)):
            lst[d] = [d] + lst[d]
        size = size
        chunks = [lst[i:i + size] for i in range(0, len(lst), size)]
        return chunks
 
    def run(self, popul):
        list_popul = popul
        req = []
        data = self.__make_chunks(list_popul, int(len(list_popul) / self.size))
        recv_data = np.zeros(shape=(self.size, len(data[0]) * 2 ), dtype=complex)
        run_score_function = 1
        for d in range(1, self.size):
            comm.send(run_score_function, dest=d)
            comm.send(data[d], dest=d)
            r = comm.irecv(buf=recv_data[d], source=d, tag=d)
            req.append(r)

        # process data corresponding to master
        # chunk 0 of list_popul
        scores = []
        for j in data[0]:
            idx = j[0]
            ind = j[1:]
            scores.append(idx)
            sc = self.cost_func.run( ind )
            scores.append(sc)
            
        recv_data[0] = scores
        #for idx in range(0, len(scores), 2):
        #    recv_data[0][idx] = idx
        #    recv_data[0][idx + 1] = scores[idx]

        all_finish = False
        while all_finish == False:
            check_finish = []
            for r in req:
                s = r.Test()
                check_finish.append(s)
            all_finish = all( [s == True for s in check_finish] )
            
        trial_scores = [1 for x in list_popul]
        #print("recv data ", recv_data)
        for l in recv_data:
            groups = [ l[i:i+2] for i in range(0, len(l), 2) ] 
            for g in groups:
                trial_scores[int(np.round(g[0].real))] = g[1].real

        #exit()
        return trial_scores
    
    def final_popul(self, popul):
        list_popul = popul
        req = []
        chunks_size = int(len(list_popul) / self.size)
        data = self.__make_chunks_popul(list_popul, chunks_size )
        recv_data = np.zeros(shape=(self.size, len(data[0]) * 3 ), dtype=complex)
        score_rmsd_function = 2
        for d in range(1, self.size):
            comm.send(score_rmsd_function, dest=d)
            comm.send(data[d], dest=d)
            r = comm.irecv(buf=recv_data[d], source=d, tag=d)
            req.append(r)

        # process data corresponding to master
        # chunk 0 of list_popul
        scores = []
        for j in data[0]:
            idx = j[0]
            ind = j[1:]
            scores.append(idx)
            sc, rmsd = self.cost_func.score_and_rmsd( ind )
            scores.append( sc )
            scores.append( rmsd )
            
        recv_data[0] = scores
        #for idx in range(0, len(scores), 2):
        #    recv_data[0][idx] = idx
        #    recv_data[0][idx + 1] = scores[idx]

        all_finish = False
        while all_finish == False:
            check_finish = []
            for r in req:
                s = r.Test()
                check_finish.append(s)
            all_finish = all( [s == True for s in check_finish] )
            
        trial_scores = [1 for x in list_popul]
        rmsd_scores = [1 for x in list_popul]
        #print("recv data ", recv_data)
        for l in recv_data:
            groups = [ l[i:i+3] for i in range(0, len(l), 3) ] 
            for g in groups:
                trial_scores[int(np.round(g[0].real))] = g[1].real
                rmsd_scores[int(np.round(g[0].real))] = g[2].real

        #exit()
        return trial_scores, rmsd_scores
 
    def terminate(self):
        # terminate
        terminate_function = 3
        for d in range(1, self.size):
            comm.send(terminate_function, dest=d)
        

class Worker():
    def __init__(self, rank, size, cost_func):
        self.rank = rank
        self.size = size
        self.cost_func = cost_func
        
    def run(self):
        functions = {1: "run_score",
                     2: "run_score_rmsd",
                     3: "terminate"}
        while (True):
            action = comm.recv(source=0)
            if (action == 3):
                break
            data = comm.recv(source=0)
            
            # process data here
            scores = [] 
            if action == 1:
                for j in data:
                    idx = j[0]
                    score_trial  = self.cost_func.run(j[1:])
                    scores.append(idx)
                    scores.append(score_trial)

            if action == 2:
                for j in data:
                    idx = j[0]
                    ind = j[1:]
                    scores.append(idx)
                    sc, rmsd = self.cost_func.score_and_rmsd( ind )
                    scores.append( sc )
                    scores.append( rmsd )
        
                    
            #print("PROCESS HERE ")
            result = scores
            send_data = np.array([result], dtype=complex) 
            comm.Send(send_data, dest=0, tag=rank)

