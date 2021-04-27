#!/mnt/netapp1/Optcesga_FT2_RHEL7/easybuild-cesga/software/Compiler/gcccore/6.4.0/python/3.7.0/bin/python3.7

import glob
import os
import shutil

prots = ["1e0m", "5wod", "1d5q"]


sources_dir = {
    "PocosReemplazos" : "./buenosResultados_PROT_3model",
    "2Fases" : "./Modelos_Folding_2Fases_1d5q",
    "4Fases" : "./Modelos_Folding_4Fases_1d5q",
    
}

def rmsd_from_model_name(name):
    return float(name.split("_")[-1].split(".")[0])

def save_models_into_folder(output_folder, models):
    for model in models:
        shutil.copyfile(model, output_folder+"/"+model.split("/")[-1])
   


def get_models_from_source(source="4Fases", prot = "1e0m", folder = "./Modelos_Folding_SOURCE_PROT"):
    output_folder = folder.replace("PROT", prot)
    output_folder = output_folder.replace("SOURCE", source)
    print(output_folder)

    # it was previously created...
    if os.path.isdir(output_folder):
        if len(output_folder) == 3:
            return output_folder
    
    source_folder = sources_dir[source].replace("PROT", prot)
    all_models = glob.glob(source_folder + "/*.pdb")
    print(all_models)
    #model_info = [(f, rmsd_from_model_name(f)) for f in all_models]
    #model_info.sort(key=lambda x: x[1])

    #best_model = model_info[0][0]
    #avg_model = model_info[int(len(model_info)/2)][0]
    #worst_model = model_info[-1][0]

    #print("selected models {} {} {} ".format(best_model, avg_model, worst_model))

    #output_folder = folder.replace("PROT", prot)
    #output_folder = output_folder.replace("SOURCE", source)
    #os.makedirs(output_folder, exist_ok=True)
    #save_models_into_folder(output_folder, [best_model, avg_model, worst_model])
    #return output_folder
    return output_folder    

def main():
    dst_1e0m = get_models_from_source("4Fases", "1e0m", "./NuevosModelos_PROT")
    dst_5wod = get_models_from_source("4Fases", "5wod", "./NuevosModelos_PROT")
    dst_1d5q = get_models_from_source("4Fases", "1d5q", "./NuevosModelos_PROT")
    print(dst_1e0m)
    print(dst_5wod)
    print(dst_1d5q)
    print(glob.glob(dst_1e0m+"/*pdb"))
    print(glob.glob(dst_5wod+"/*pdb"))
    
if __name__== "__main__":
    main()

