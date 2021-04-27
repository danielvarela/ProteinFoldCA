#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import glob
import os
import shutil
import sys

main_path = "/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta/ANN_folding/"
experiments = ["unfolded", "3models", "3modelsAndStraight"]
# experiments = ["unfolded", "3modelsAndStraight"]
experiments = ["3modelsAndStraight"]
prots = ["5wod"]

rosetta_output_2Fases = {
    #"1e0m": "./Rosetta2Fases_PDBS_1e0m/",
    "1e0m": "./starting_point/",
    "5wod": "./starting_point_5wod/"
    #"5wod": "./Rosetta2Fases_PDBS_5wod/"
}


rosetta_output_Pocos = {
    "1e0m": "./RosettaPOCOS_1e0m/",
    "5wod": "./RosettaPOCOS_5wod/"
}

def print_file(filename, prot,  jobs):
    with open("./"+filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -n {}\n".format(len(jobs)))
        f.write("#SBATCH --job-name=folding_{}\n".format(prot))
        f.write("#SBATCH -a 1\n")
        f.write("#SBATCH -t 0:15:00\n")
        f.write("#SBATCH -p cola-corta,thinnodes\n")
        # f.write("#SBATCH -p shared\n")
        # f.write("#SBATCH --qos shared\n")
        f.write("#SBATCH -o 'ann-%x-%A_%a.out'\n")
        f.write("#SBATCH -e 'error-%x.out'\n")
        # f.write("#SBATCH --mem-per-cpu=5GB\n")
        f.write("#SBATCH --export=job_id='%A_%a'\n")
        f.write("\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/opt/cesga/easybuild-cesga/software/MPI/gcc/6.4.0/openmpi/2.1.1/mpi4py/3.0.2-python-3.7.0/lib/python3.7/site-packages/\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/mnt/netapp2/Home_FT2/home/ulc/co/dvm/.local/lib/python3.7/site-packages/\n")
        f.write("module load python/3.7.0\n")
        f.write("module load cesga/2018\n")
        f.write("module load gcc/6.4.0\n")
        f.write("module load openmpi/2.1.1\n")
        f.write("module load mpi4py/3.0.2-python-3.7.0\n")
        f.write("\n")
        for j in jobs: 
            f.write("cd {}\n".format(j.split("config")[0].replace("./", main_path)))
            f.write("../../../../../perform_folding_at_pdbs.py {} &\n".format(j.split("/")[-1]))
        f.write("wait\n")


def get_job(prot, experiment, list_bests):
    job_list = ["unfolded", "3models", "3modelsAndStraight"]
    with open(list_bests, "r") as flist:
        bests = [l.strip() for l in flist.readlines()]
    by_prot = [e for e in bests if prot in e]
    by_experiment = ""
    for p in by_prot:
        for opt in p.split("/"):
            if ("_") in opt:
                words = opt.split("_")
                if experiment in words:
                    by_experiment = p
                    break
    # by_experiment = [e for e in by_prot if experiment in [d.split("_") for d in e.split("/") if any(ext in d for ext in job_list)]]
    if len(by_experiment) == 0:
        print("job not found")
        exit()
    else:
        selected_job = by_experiment
        selected_job = selected_job.replace("evolution", "ann_job")
        if not os.path.isfile(selected_job):
            print("selected job {} not exists ".format(selected_job))
            exit()
    return selected_job


list_bests_paths = {
    "Pocos" : "./lists_bests_PocosReemplazos.txt",
    "Fase2" : "./lists_bests_Fase2.txt",
    "Fase4" : "./list_for_Resultados4Fases_1e0m.txt",
}

ross_input = sys.argv[-1]

list_bests = list_bests_paths[ross_input]


if ross_input == "Fase2":
    source = rosetta_output_2Fases
elif ross_input == "Pocos":
    source = rosetta_output_Pocos
elif ross_input == "Fase4":
    print("pendiente de hacer la lista")
    exit()

    
for p in prots:
    all_files = glob.glob(source[p] + "/*.pdb")
    total_len = len(all_files)
    ncores = 1
    n = int(total_len / ncores)
    chunks = [all_files[i:i + n] for i in range(0, len(all_files), n)]
    
    salida = "PDBs" 
    output_folder = "./"+salida+"/"+ross_input+"/"+p+"/"
    os.makedirs(output_folder, exist_ok=True)
        
    sources = []
    for i, chunk in enumerate(chunks):
        output_folder_pdb = "./"+salida+"/"+ross_input+"/"+p+"/pdbs/"+str(i) 
        os.makedirs(output_folder_pdb, exist_ok=True) 
        for pdb in chunk:
            shutil.copyfile(pdb, output_folder_pdb +"/"+ pdb.split("/")[-1])
        sources.append(output_folder_pdb)


    for exp in experiments:
        jobs = []
        input_operator = get_job(p, exp, list_bests)
        for i, source in enumerate(sources):
            ofolder = output_folder + "/"+exp+"/" +str(i)
            os.makedirs(ofolder, exist_ok=True)
            if not os.path.exists(ofolder + "/data_pdbs"):
                os.symlink(main_path + "data_pdbs", ofolder + "/data_pdbs")
            name = ofolder + "/config_file_"+str(i)+".ini"
            with open(name, "w") as f:
                f.write("[info]\n")
                f.write("operator = {}\n".format(input_operator.replace("./", main_path)))
                f.write("prot = {}\n".format(p))
                f.write("experiment = {}\n".format(exp))
                f.write("source_pdb = ../../pdbs/{}\n".format(i))
            jobs.append(name)
        sbatch_name = "folding_job_"+ross_input+"_"+exp+"_"+p+".sh"
        print_file(sbatch_name, p, jobs)
        print("sbatch {}".format(sbatch_name))
    # for j in jobs: 
    #    print("cd {}".format(j.split("config")[0].replace("./", main_path)))
    #    print("../../../../../perform_folding_at_pdbs.py {}".format(j.split("/")[-1]))
    exit()
