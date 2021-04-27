#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import glob
import os
import shutil
import sys


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

structures_source = "4Fases"

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
    

def print_folding_job_file(output_folder, jobs):
    filename = "/run_jobs.sh"
    print(jobs)
    with open(output_folder + filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -n {}\n".format(len(jobs)))
        f.write("#SBATCH --job-name=folding\n")
        f.write("#SBATCH -a 1\n")
        f.write("#SBATCH -t 10:00:00\n")
        f.write("#SBATCH -p shared\n")
        f.write("#SBATCH --qos shared\n")
        f.write("#SBATCH -o 'ann-%x-%A_%a.out'\n")
        f.write("#SBATCH -e 'error-%x.out'\n")
        f.write("#SBATCH --mem-per-cpu=2GB\n")
        f.write("#SBATCH --export=job_id='%A_%a'\n")
        f.write("\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta/\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/opt/cesga/easybuild-cesga/software/MPI/gcc/6.4.0/openmpi/2.1.1/mpi4py/3.0.2-python-3.7.0/lib/python3.7/site-packages/\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/mnt/netapp2/Home_FT2/home/ulc/co/dvm/.local/lib/python3.7/site-packages/n")
        f.write("module load python/3.7.0\n")
        f.write("module load cesga/2018\n")
        f.write("module load gcc/6.4.0\n")
        f.write("module load openmpi/2.1.1\n")
        f.write("module load mpi4py/3.0.2-python-3.7.0\n")
        #f.write("source ../vfolding/bin/activate\n")
        f.write("cd /mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta/ANN_folding\n\n")
        for j in jobs:
            f.write("srun {} &\n".format(j))
        f.write("wait\n")


def ini_file(operator, prot, exp):
    os.makedirs("folding_ini_files", exist_ok=True)
    ini_file_name = "folding_ini_files/{}_{}.ini".format(prot, exp)
    with open(ini_file_name, "w") as f:
        f.write("[info]\n")
        f.write("operator={}\n".format(operator))
        f.write("prot={}\n".format(prot))
        f.write("experiment={}\n".format(exp))
        if structures_source == "2Fases": 
            f.write("source_pdb={}\n".format(ROSETTA_STRUCTURE_STAGE2[prot]))
        else:
            f.write("source_pdb={}\n".format(ROSETTA_STRUCTURE_STAGE4[prot]))
    return ini_file_name
    
def params(operator, prot, exp):
    if structures_source == "2Fases": 
        return "{} {} {} {}".format(operator, prot, exp, ROSETTA_STRUCTURE_STAGE2[prot])
    else:
        return "{} {} {} {}".format(operator, prot, exp, ROSETTA_STRUCTURE_STAGE4[prot])
 
def main():
    list_name = "list_for_structures{}.txt".format(structures_source)
    #input_experiment = sys.argv[-1]
    #result_folder = glob.glob(input_experiment + "/*")
    data = {}
    # for exp in experiments:
    #     path = [r for r in result_folder if exp in r]
    #     path = path[0]
    #     data[exp] = path

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

        

    with open(list_name, "w") as f:
        for exp, path in bst_data.items():
            f.write("./"+path+"\n")

    

    fold_run = "./fold_srun"
    os.makedirs(fold_run, exist_ok=True)

    pyscript = "./perform_folding_at_pdbs.py"
    my_path = "/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta/ANN_folding/"
    jobs = []
    for exp, path in bst_data.items():
        prot = path.split("/")[0].split("_")[-1]
        #prot = "1e0m" if "1e0m" in input_experiment else "5wod"
        srun = fold_run + "/exec_"+list_name.replace(".txt","")+"_"+exp+"_"+prot+".sh"
        with open(srun, "w") as f:
            f.write("#!/bin/bash\n")
            #cmd = "{} {}{} {} {}".format(pyscript,my_path,list_name, prot, exp)
            cmd = "{} {}".format(pyscript,params(path, prot, exp))
            f.write(cmd+"\n")
        jobs.append(srun)

    print_folding_job_file(fold_run, jobs)
    print("chmod +x ./fold_srun/*.sh && sbatch ", fold_run + "/run_jobs.sh")
    os.system("chmod +x ./fold_srun/*.sh && sbatch  ./fold_srun/run_jobs.sh")

    
if __name__== "__main__":
    main()

