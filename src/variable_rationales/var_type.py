import json
import numpy as np
import os
from tqdm import trange
import torch
from torch import nn

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)



def select_between_types_(data_split_name,model_random_seed):

    print('===============')
    print(' starting select between types (inside)')

    desc = f'selecting between rationale types -> {data_split_name}'

    ## load topk rationale_masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "topk",
        ""
    )

    ## check if we have topk rationales
    if os.path.exists(fname + data_split_name + "-rationale_metadata.npy"):

        print( fname + data_split_name + "-rationale_metadata.npy EXIT !")

        topk_rationales = np.load(fname + data_split_name + "-rationale_metadata.npy", allow_pickle = True).item()
        print(f"*** loaded succesfully -> {fname}{data_split_name}-rationale_metadata.npy")

    else:

        print(f"{fname}{data_split_name}-rationale_metadata.npy DOES NOT EXIST")
        print("** returning to main -- RATIONALE TYPES NOT SELECTED")

        return

    ## load contigious rationale_masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "contigious",
        ""
    )

    ## check if we have contigious rationales
    if os.path.exists(fname + data_split_name + "-rationale_metadata.npy"):

        cont_rationales = np.load(fname + data_split_name + "-rationale_metadata.npy", allow_pickle = True).item()
        print(f"*** loaded succesfully -> {fname}{data_split_name}-rationale_metadata.npy")

    else:

        print(f"{fname}{data_split_name}-rationale_metadata.npy DOES NOT EXIST")
        print("** returning to main -- RATIONALE TYPES NOT SELECTED")

        return

    pbar = trange(len(cont_rationales.keys()), desc = desc)

    for annot_id in cont_rationales.keys():
        

        ## variable
        contig = cont_rationales[annot_id]["var-len_var-feat"]
        topk = topk_rationales[annot_id]["var-len_var-feat"]


        if contig["variable-length divergence"] > topk["variable-length divergence"]:

            topk_rationales[annot_id]["var-len_var-feat_var-type"] =  cont_rationales[annot_id]["var-len_var-feat"]
            cont_rationales[annot_id]["var-len_var-feat_var-type"] =  cont_rationales[annot_id]["var-len_var-feat"]
        
        else:

            topk_rationales[annot_id]["var-len_var-feat_var-type"] =  topk_rationales[annot_id]["var-len_var-feat"]
            cont_rationales[annot_id]["var-len_var-feat_var-type"] =  topk_rationales[annot_id]["var-len_var-feat"]

        ## fixed
        contig = cont_rationales[annot_id]["fixed-len_var-feat"]
        topk = topk_rationales[annot_id]["fixed-len_var-feat"]

        if contig["fixed-length divergence"] > topk["fixed-length divergence"]:

            topk_rationales[annot_id]["fixed-len_var-feat_var-type"] =  cont_rationales[annot_id]["fixed-len_var-feat"]
            cont_rationales[annot_id]["fixed-len_var-feat_var-type"] =  cont_rationales[annot_id]["fixed-len_var-feat"]

        else:

            topk_rationales[annot_id]["fixed-len_var-feat_var-type"] =  topk_rationales[annot_id]["fixed-len_var-feat"]
            cont_rationales[annot_id]["fixed-len_var-feat_var-type"] =  topk_rationales[annot_id]["fixed-len_var-feat"]

        pbar.update(1)


    ## save data
    ## save topk rationale_masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "topk",
        ""
    )
    import random
    random_id = random.choice(list(topk_rationales.keys()))

    print('  ################# ')
    if 'fixed-len_var-feat_var-type' in topk_rationales[random_id].keys():
        print('###### - fixed-len_var-feat_var-type -  exit')
    else:
        print('###### - fixed-len_var-feat_var-type -  NOT exit')
    np.save(fname + data_split_name + "-rationale_metadata.npy", topk_rationales)

    ## save contigious rationale_masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "contigious",
        ""
    )

    np.save(fname + data_split_name + "-rationale_metadata.npy", cont_rationales)
    

    return

            