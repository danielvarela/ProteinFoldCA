#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import glob
import os
import shutil
import sys
import pandas as pd


# experiments = ["unfolded", "3models", "3modelsAndStraight"]
# experiments = ["3models"]

experiments = ["unfolded_{}", "3models_4Fases_{}", "3modelsAndStraight_4Fases_{}"]

ROSETTA_STRUCTURE_STAGE2 = {
    "1d5q": "RosettaAbinitio_2Fases_1d5q",
    "5wod" : "Rosetta2Fases_PDBS_5wod",
    "1e0m" : "Rosetta2Fases_PDBS_1e0m",
}
ROSETTA_STRUCTURE_STAGE4 = {
    "1d5q": "RosettaAbinitio_4Fases_1d5q",
    "1e0m" : "Rosetta4Fases_PDBS_1e0m",
    "5wod" : "Rosetta4Fases_PDBS_5wod",
    
}

structures_source = "2Fases"

def find_best_at_evolutions(evolutions):
    current_best_rmsd = 1000
    results = []
    for filepath in evolutions:
        with open(filepath, "r") as f:
            rmsds = f.readlines()[-1]
            if "RMSDS: " in rmsds:
                best_rmsds = rmsds.strip().replace("RMSDS: ", "")[:-1].split(",")
                best_rmsds = [float(v) for v in best_rmsds]
                best_rmsd = min(best_rmsds)
                results.append((best_rmsd, filepath))
                if best_rmsd < current_best_rmsd:
                    current_best_rmsd = best_rmsd
                    best_rmsd_path = filepath

    results.sort(key=lambda x: x[0])
    best_rmsd_path = results[0][1]
    return best_rmsd_path 
    
   
def find_best_refinement(prot, exp):
    csv = "resultados_folding_3proteins/results_folding_Rosetta{}_{}_{}.csv".format(structures_source, exp,prot)
    df = pd.read_csv(csv)
    df.sort_values("rmsd", ascending=True, inplace=True)

    path_pdb = df.iloc[0]["id"]
    os.makedirs("run_operator", exist_ok=True)
    os.system("cp {} run_operator/".format(path_pdb))
    #print(df.iloc[0])
    return path_pdb
    
def params(operator, prot, exp):
    if structures_source == "2Fases": 
        return "{} {} {} {}".format(operator, prot, exp, ROSETTA_STRUCTURE_STAGE2[prot])
    else:
        return "{} {} {} {}".format(operator, prot, exp, ROSETTA_STRUCTURE_STAGE4[prot])
 
def main():
    data = {}
    prot = sys.argv[-1]
    exps = [p.format(prot) for p in experiments]
    for experiment in exps:
        exp = experiment.split("_")[0]
        path = experiment
        data[exp] = path
    bst_data = {}
    for exp, path in data.items():
        evolutions = glob.glob(path + "/*evolution*txt")
        best_trained = find_best_at_evolutions(evolutions)
        best_ann = best_trained.replace("evolution","ann_job")
        bst_data[exp] = best_ann

   
    pyscript = "run_operator_on_pdb.py"
    my_path = "/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta/ANN_folding/"
    jobs = []
    for exp, path in bst_data.items():
        prot = path.split("/")[0].split("_")[-1]
        find_best_refinement(prot, exp)
        cmd = "python {} {} {}".format(pyscript,path, prot)
        print(cmd+"\n")
        os.system(cmd)
        name_source = "STR_snapshots_pose_Rosetta{}_{}_{}".format(structures_source, exp,prot)
        os.makedirs("{}".format(name_source), exist_ok=True)
        os.system("cp final_pose*pdb {}".format(name_source))
        os.system("rm -rf run_operator/*pdb")


    
if __name__== "__main__":
    main()

