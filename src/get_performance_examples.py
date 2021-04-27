#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import glob
import os
import shutil
import sys

def main():
    fold_run = "./fold_srun"

    exps = []
    for fname in glob.glob(fold_run + "/exec_*"): 
        prot = fname.split("_")[-1].replace(".sh", "")
        exp = fname.split("_")[-2]
        fases = fname.split("_")[-3]
        exps.append({"prot" : prot, "exp" : exp, "fname" : fname, "fases": fases})

    for e in exps:
        outpath = "ejemplos_performance_PartiallyFolded/"+e["prot"]+"/"+e["exp"] + "/" + e["fases"]
        os.makedirs(outpath, exist_ok=True)
        with open(e["fname"]) as f:
            run = [x.rstrip() for x in f.readlines() if len(x.rstrip()) > 1][-1]
            print(run)
            os.system(run)
            os.system("cp "+ run.split(" ")[1] + " "+ outpath)
            os.system("cp performance_ann_1.txt " + outpath)
            
if __name__== "__main__":
    main()

