import torch
from torch import nn
import json
from tqdm import trange
import numpy as np
import pandas as pd
import config.cfg
from config.cfg import AttrDict
import os

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nn.deterministic = True
torch.backends.cudnn.benchmark = False
    

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)

from src.evaluation import thresholders
from src.common_code.useful_functions import wpiece2word 

def rationale_creator_(data, data_split_name, variable, tokenizer):

    ## get the thresholder fun
    thresholder = getattr(thresholders, args["thresholder"])

    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        ""
    )

    fname += data_split_name + "-rationale_metadata.npy"

    ## retrieve importance scores
    rationale_metadata = np.load(fname, allow_pickle = True).item()

    ## filter only relevant parts in our dataset
    data = data[["text", "annotation_id", "exp_split", "label", "label_id"]]

    data["tokenized_text"] = data["text"]

    annotation_text = dict(data[["annotation_id", "tokenized_text"]].values)

    del data["tokenized_text"]

    ## time to register rationales
    for feature_attribution in {"attention", "gradients", "ig", "scaled attention"}:
        

        temp_registry = {}

        for annotation_id, sequence_text in annotation_text.items():
            

            ## check if there is any padding which could affect our process and remove
            seq_length = (np.asarray(sequence_text) != 0).sum()

            sequence_importance = rationale_metadata[annotation_id][feature_attribution]["importance scores"][:seq_length]
            sequence_text = sequence_text[:len(sequence_importance)]

            if variable:

                # untokenize sequence and sequence importance scores
                sequence_text, sequence_importance = wpiece2word(
                    tokenizer = tokenizer, 
                    sentence = sequence_text, 
                    weights = sequence_importance
                )

                rationale_indxs = thresholder(
                    scores = sequence_importance, 
                    original_length = len(sequence_text) - 2,
                    rationale_length =  rationale_metadata[annotation_id][feature_attribution]["variable rationale ratio"]
                )

            else:

                ## untokenize sequence and sequence importance scores
                sequence_text, sequence_importance = wpiece2word(
                    tokenizer = tokenizer, 
                    sentence = sequence_text, 
                    weights = sequence_importance
                )

                rationale_indxs = thresholder(
                    scores = sequence_importance, 
                    original_length = len(sequence_text) -2,
                    rationale_length = args["rationale_length"]
                )

            rationale = sequence_text[rationale_indxs]

            temp_registry[annotation_id] = " ".join(rationale)

        data.text = data.annotation_id.apply(
            lambda x : temp_registry[x]
        )

        fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            args["thresholder"],
            "data",
            ""
        )

        os.makedirs(fname, exist_ok=True)

        
        if variable:
            fname += "var_len-" + feature_attribution + "-" + data_split_name + ".csv"
        else:
            fname += feature_attribution + "-" + data_split_name + ".csv"

        print(f"saved in -> {fname}")

        data.to_csv(fname)

    ## now to save our (fixed-len + var-feat) and (var-len + var-feat rationales)

    temp_registry = {}

    if variable: for_our_approach = "var-len_var-feat"
    else: for_our_approach = "fixed-len_var-feat"

    for annotation_id, sequence_text in annotation_text.items():
        

        ## check if there is any padding which could affect our process and remove
        seq_length = (np.asarray(sequence_text) != 0).sum()

        sequence_importance = rationale_metadata[annotation_id][for_our_approach]["importance scores"][:seq_length]
        sequence_text = sequence_text[:len(sequence_importance)]

        # untokenize sequence and sequence importance scores
        sequence_text, sequence_importance = wpiece2word(
            tokenizer = tokenizer, 
            sentence = sequence_text, 
            weights = sequence_importance
        )

        rationale_indxs = thresholder(
            scores = sequence_importance, 
            original_length = len(sequence_text) - 2,
            rationale_length =  rationale_metadata[annotation_id][for_our_approach]["variable rationale ratio"]
        )

        rationale = sequence_text[rationale_indxs]

        temp_registry[annotation_id] = " ".join(rationale)

    data.text = data.annotation_id.apply(
        lambda x : temp_registry[x]
    )

    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        "data",
        ""
    )

    os.makedirs(fname, exist_ok=True)

    
    if variable:
        fname += "var-len_var-feat-" + data_split_name + ".csv"
    else:
        fname += "fixed-len_var-feat-" + "-" + data_split_name + ".csv"

    print(f"saved in -> {fname}")

    data.to_csv(fname)


    return
