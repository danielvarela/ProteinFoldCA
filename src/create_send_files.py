#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import glob
import os
import shutil

prots = ["1e0m", "5wod"]
experiments = ["unfolded", "3models", "3modelsAndStraight"]

native_paths = {"5wod": "./data_pdbs/5WOD/5wod_A.pdb",
                "1e0m": "./data_pdbs/1E0M/1e0m_A.pdb"}

def print_file(filename, prot, exp, output_folder):
    native = native_paths[prot]
    with open(output_folder + filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -n 24\n")
        f.write("#SBATCH --job-name=steps20_{}\n".format(prot))
        f.write("#SBATCH -a 1-25\n")
        f.write("#SBATCH -t 4:30:00\n")
        f.write("#SBATCH -p cola-corta,thinnodes\n")
        f.write("#SBATCH -o 'ann-%x-%A_%a.out'\n")
        f.write("#SBATCH -e 'error-%x.out'\n")
        f.write("#SBATCH --mem-per-cpu=1GB\n")
        f.write("#SBATCH --export=job_id='%A_%a'\n")
        f.write("\n")
        f.write("expot PYTHONPATH=$PYTHONPATH:/mnt/netapp2/Store_uni/home/ulc/co/dvm/PyRosetta\n")
        f.write("export PYTHONPATH=$PYTHONPATH:/opt/cesga/easybuild-cesga/software/MPI/gcc/6.4.0/openmpi/2.1.1/mpi4py/3.0.2-python-3.7.0/lib/python3.7/site-packages/\n")
        f.write("module load python/3.7.0\n")
        f.write("module load cesga/2018\n")
        f.write("module load gcc/6.4.0\n")
        f.write("module load openmpi/2.1.1\n")
        f.write("module load mpi4py/3.0.2-python-3.7.0\n")
        f.write("srun ./my_rosetta.py {} {}\n".format(native, exp))

def clean_output(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    has_files = glob.glob(output_folder+"*.sh")
    print(has_files)
    if len(has_files) > 1:
        print("output folder {} has files".format(output_folder))
        delete = input(" delete [1/0]? > ")
        if delete == 1:
            for f in has_files:
                os.remove(f)
        
def main():
    created = []
    OUTPUT_FOLDER = "./sbatch/"
    clean_output(OUTPUT_FOLDER)
    for prot in prots:
        for exp in experiments:
                filename = "job_{}_{}.sh".format(exp, prot)
                print_file(filename, prot, exp, OUTPUT_FOLDER)
                created.append(filename)
 
    print("files")
    for f in created:
        print("sbatch {} ".format(OUTPUT_FOLDER + f))
                
if __name__== "__main__":
    main()

