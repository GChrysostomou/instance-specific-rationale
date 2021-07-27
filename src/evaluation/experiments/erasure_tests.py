import torch
import torch.nn as nn
import math 
import json
from tqdm import trange
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

from src.common_code.useful_functions import batch_from_dict
from src.common_code.metrics import normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_

from sklearn.metrics import classification_report

def conduct_tests_(model, data, model_random_seed):

    """
        Info: computes the average fraction of tokens required to cause a decision flip (prediction change)
        Input:
            model : pretrained model
            data : torch.DataLoader loaded data
            save_path : path to save the results
        Output:
            saves the results to a csv file under the save_path
    """

    desc = 'conducting faithfulness tests'
    
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)

    ## load rationale masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        "test-rationale_metadata.npy"
    )

     ## retrieve importance scores
    rationale_metadata = np.load(fname, allow_pickle = True).item()

    faithfulness_scores = {}

    for batch in data:
        
        model.eval()
        model.zero_grad()

        batch = [torch.stack(t).transpose(0,1) if type(t) is list else t for t in batch]
        
        inputs = {
            "sentences" : batch[0].to(device),
            "lengths" : batch[1].to(device),
            "labels" : batch[2].to(device),
            "annotation_id" : batch[3],
            "query_mask" : batch[4].to(device),
            "token_type_ids" : batch[5].to(device),
            "attention_mask" : batch[6].to(device),
            "retain_gradient" : True
        }
                                        
        assert inputs["sentences"].size(0) == len(inputs["labels"]), "Error: batch size for item 1 not in correct position"

        ## retrieve original full text logits and converts to probs
        original_prediction = batch_from_dict(
            batch_data = inputs, 
            rationale_data = rationale_metadata, 
            target_key = "original prediction", 
            feature_attribution = None
        )

        ## saving our results
        for annotation_id in inputs["annotation_id"]:

            faithfulness_scores[annotation_id] = {}

        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy()

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        original_sentences = inputs["sentences"].clone()

        rows = np.arange(original_sentences.size(0))

        ## now measuring baseline sufficiency for all 0 rationale mask
        inputs["sentences"] = torch.zeros_like(original_sentences)

        yhat, _  = model(**inputs)

        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        with torch.no_grad():

            ## start the masking process to measure faithfulness
            for length_of_rationale in {"fixed", "variable"}:
                
                ## alias for the variable stuff and for saving
                if length_of_rationale == "variable":

                    var_alias = "var"

                else:

                    var_alias = "fixed"

                for feat_attribution_name in {"attention", "ig", "gradients", "scaled attention", "random", "--var-feat"}:

                        if feat_attribution_name == "--var-feat":

                            feat_attribution_name = var_alias + "-len_var-feat"

                        rationale_mask = batch_from_dict(
                            inputs, 
                            rationale_metadata, 
                            feature_attribution = feat_attribution_name, 
                            target_key = f"{length_of_rationale} rationale mask"
                        )

                        ## measuring faithfulness
                        comp, reduced_probs  = normalized_comprehensiveness_(
                            model = model, 
                            original_sentences = original_sentences, 
                            rationale_mask = rationale_mask, 
                            inputs = inputs, 
                            full_text_probs = full_text_probs, 
                            full_text_class = full_text_class, 
                            rows = rows,
                            suff_y_zero = suff_y_zero
                        )

                        suff = normalized_sufficiency_(
                            model = model, 
                            original_sentences = original_sentences, 
                            rationale_mask = rationale_mask, 
                            inputs = inputs, 
                            full_text_probs = full_text_probs, 
                            full_text_class = full_text_class, 
                            rows = rows,
                            suff_y_zero = suff_y_zero
                        )

                        for j_, annot_id in enumerate(inputs["annotation_id"]):
                            
                            if feat_attribution_name == "--var-feat":

                                dic_name = feat_attribution_name

                            else:

                                dic_name = var_alias + "-" + feat_attribution_name

                            faithfulness_scores[annot_id][dic_name] = {
                                "comprehensiveness" : float(comp[j_]),
                                "sufficiency" : float(suff[j_]),
                                "masked prediction probs" : reduced_probs[j_].astype(np.float64).tolist(),
                                "full text prediction probs" : original_prediction[j_].astype(np.float64).tolist(),
                                "labels" : int(inputs["labels"][j_].detach().cpu().item())
                            }

                    
        pbar.update(data.batch_size)


    ## save our results
    fname = args["evaluation_dir"] + args.thresholder + "-faithfulness-metrics.json"

    with open(fname, 'w') as file:
            json.dump(
                faithfulness_scores,
                file,
                indent = 4
            ) 

    averages = {}
    f1s_model = {
        "f1 macro avg - model labels" : {},
        "f1 macro avg - actual labels" : {}
    }

    random_annot_id = inputs["annotation_id"][0]

    feat_attributions = faithfulness_scores[random_annot_id].keys()

    for key in feat_attributions:

            averages[key] = {}

            for metric in {"comprehensiveness", "sufficiency"}:

                averages[key][metric] = {
                    "mean" : np.asarray([v[key][metric] for k,v in faithfulness_scores.items()]).mean(),
                    "std" : np.asarray([v[key][metric] for k,v in faithfulness_scores.items()]).std()
                } 


            masked_prediction_probs = [np.asarray(faithfulness_scores[k][key]["masked prediction probs"]).argmax() for k in faithfulness_scores.keys()]
            full_text__prediction_probs = [np.asarray(faithfulness_scores[k][key]["full text prediction probs"]).argmax() for k in faithfulness_scores.keys()]
            true_labels = [faithfulness_scores[k][key]["labels"] for k in faithfulness_scores.keys()]

            out_model = classification_report(full_text__prediction_probs, masked_prediction_probs, output_dict = True)["macro avg"]["f1-score"]
            out_labels = classification_report(true_labels, masked_prediction_probs, output_dict = True)["macro avg"]["f1-score"]

            f1s_model["f1 macro avg - model labels"][key] = round(out_model * 100, 3)
            f1s_model["f1 macro avg - actual labels"][key] = round(out_labels * 100, 3)

    ## save descriptors
    fname = args["evaluation_dir"] + args.thresholder + "-faithfulness-metrics-description.json"


    with open(fname, 'w') as file:
            json.dump(
                averages,
                file,
                indent = 4
            ) 

    ## save f1s
    fname = args["evaluation_dir"] + args.thresholder + "-f1-metrics-description.json"


    with open(fname, 'w') as file:
            json.dump(
                f1s_model,
                file,
                indent = 4
            ) 

    return
