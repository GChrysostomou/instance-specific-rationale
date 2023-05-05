import torch
import torch.nn as nn
import math 
import json
import numpy as np
import os

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

from sklearn.metrics import classification_report

def compute_faithfulness_(rationale_metadata, prediction_data, split_name):

    metric_combos = [
        {"gradients"},
        {"lime"}, 
        {"attention"},
        {"gradients", "ig"}, 
        {"scaled attention", "attention"}, 
        {"scaled attention", "lime"},
        {"deeplift", "attention"},  
        {"gradients", "ig", "deeplift"},
        {"attention", "scaled attention", "lime"},
        {"deeplift", "scaled attention", "lime"},
        {"gradients", "ig", "deeplift", "attention"},
        {"gradients", "lime", "deeplift", "scaled attention"},
        {"gradients", "ig", "deeplift", "attention", "scaled attention"},
        {"gradients", "ig", "deeplift", "attention", "lime"},
        {"gradients", "ig", "deeplift", "attention", "scaled attention", "lime"}
    ]        
    
    faith_increasing = {}

    ## lets check if with increasing combos
    ## we obtain better performance
    for what_kind in {"fixed", "variable"}:
        
        faith_increasing[what_kind] = {}

        for metric_combo in metric_combos:
            
            ## temporary scorers of information
            predicted_labels = []
            full_text_labels = []
            actual_labels = []
            sufficiencies = []
            comprehensiveness = []

            faith_increasing[what_kind]["-".join(metric_combo)] = {}

            for annotation_id in rationale_metadata.keys():
                
                ## initiators
                init_div = float("-inf")

                for feat_name in metric_combo:
                    
                    div = rationale_metadata[annotation_id][feat_name][f"{what_kind}-length divergence"]

                    if div > init_div:

                        if what_kind == "variable":

                            alias = "var"

                        else: 

                            alias = what_kind
                        print(annotation_id)
                        sufficiency = prediction_data[annotation_id][f"{alias}-{feat_name}"]["sufficiency"]
                        comprehensive = prediction_data[annotation_id][f"{alias}-{feat_name}"]["comprehensiveness"]
                        predicted_lab = np.asarray(prediction_data[annotation_id][f"{alias}-{feat_name}"]["masked prediction probs"]).argmax()
                        full_text_lab = np.asarray(prediction_data[annotation_id][f"{alias}-{feat_name}"]["full text prediction probs"]).argmax()
                        actual_lab = prediction_data[annotation_id][f"{alias}-{feat_name}"]["labels"]

                        init_div = div

                actual_labels.append(actual_lab)
                full_text_labels.append(full_text_lab)
                predicted_labels.append(predicted_lab)
                sufficiencies.append(sufficiency)
                comprehensiveness.append(comprehensive)

            model_score = classification_report(full_text_labels, predicted_labels, output_dict = True)["macro avg"]["f1-score"]
            label_score = classification_report(actual_labels, predicted_labels, output_dict = True)["macro avg"]["f1-score"]

            sufficiencies = np.asarray(sufficiencies)
            comprehensiveness = np.asarray(comprehensiveness)

            faith_increasing[what_kind]["-".join(metric_combo)] = {
                "f1 score - model labels" : round(model_score * 100, 3),
                "f1 score - actual labels" : round(label_score * 100, 3),
                "sufficiency" : {
                    "mean" :  sufficiencies.mean(),
                    "std"  :  sufficiencies.std()
                },
                "comprehensiveness" : {
                    "mean" :  comprehensiveness.mean(),
                    "std"  :  comprehensiveness.std()
                }
            }

    ## save descriptors
    fname = args["evaluation_dir"] + args.thresholder + f"-{split_name}-increasing-feature-scoring.json"


    with open(fname, 'w') as file:
            json.dump(
                faith_increasing,
                file,
                indent = 4
            ) 

    return
