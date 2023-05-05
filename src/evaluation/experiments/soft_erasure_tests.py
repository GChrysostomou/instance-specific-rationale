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

from src.common_code.useful_functions import create_only_query_mask_
from src.common_code.metrics import normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_

from sklearn.metrics import classification_report


def comprehensiveness_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)

    return comprehensiveness

    
def soft_conduct_tests_(model, data, split, model_random_seed):

    """
        Info: computes the average fraction of tokens required to cause a decision flip (prediction change)
        Input:
            model : pretrained model
            data : torch.DataLoader loaded data
            save_path : path to save the results
        Output:
            saves the results to a csv file under the save_path
    """

    desc = f'conducting faithfulness tests on -> {split}'
    
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)

    ## load rationale masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        f"{split}-rationale_metadata.npy"
    )
    print('load rationale masks at: ', fname)
    
    ## retrieve importance scores
    rationale_metadata = np.load(fname, allow_pickle = True).item()
    rationale_metadata_keys_list = list(rationale_metadata.keys())
    print('one example for retrieve importance scores: ', rationale_metadata.get(rationale_metadata_keys_list[2])) #rationale_metadata is a dictionary

    faithfulness_scores = {}

    for i, batch in enumerate(data):
        if i == 0:
            print('the first batch in data for reference: ')
            print(batch)
        model.eval()
        model.zero_grad()

        batch = {
                "annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "query_mask" : batch["query_mask"].squeeze(1).to(device),
                "special_tokens" : batch["special tokens"],
                "retain_gradient" : False ## we do not need it
            }
            
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
        yhat, _ =  model(**batch)

        ## retrieve original full text logits and converts to probs
        original_prediction = batch_from_dict(
            batch_data = batch,
            metadata = rationale_metadata,
            target_key =  "original prediction",
            extra_layer = None
        )

        ## saving our results
        for annotation_id in batch["annotation_id"]:

            faithfulness_scores[annotation_id] = {}

        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy()

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        original_sentences = batch["input_ids"].clone()

        rows = np.arange(original_sentences.size(0))

        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:

            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"]
            )

            batch["input_ids"] = only_query_mask * original_sentences

        else:

            only_query_mask=torch.zeros_like(batch["input_ids"]).long()

            batch["input_ids"] = only_query_mask

        yhat, _  = model(**batch)

        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        with torch.no_grad():

            ## start the masking process to measure faithfulness
            for length_of_rationale in ["fixed"]: #, "variable"
                
                ## alias for the variable stuff and for saving
                if length_of_rationale == "variable":

                    var_alias = "var"

                else:

                    var_alias = "fixed"

                for feat_attribution_name in ["deeplift","lime", "attention", "ig", "gradients", "scaled attention", "random"]: #, "--var-feat", "--var-all"
                        if feat_attribution_name == "--var-all":
    
                            feat_attribution_name = var_alias + "-len_var-feat_var-type"
                        
                        if feat_attribution_name == "--var-feat":

                            feat_attribution_name = var_alias + "-len_var-feat"

                        rationale_mask = batch_from_dict_(
                            batch_data = batch,
                            metadata = rationale_metadata,
                            target_key =  f"{length_of_rationale} rationale mask",
                            extra_layer = feat_attribution_name,
                        ) # bug

                        ## measuring faithfulness
                        ## for soft-comprehensiveness, change this
                        comp, reduced_probs  = normalized_comprehensiveness_(
                            model = model, 
                            original_sentences = original_sentences, 
                            rationale_mask = rationale_mask, 
                            inputs = batch, 
                            full_text_probs = full_text_probs, 
                            full_text_class = full_text_class, 
                            rows = rows,
                            suff_y_zero = suff_y_zero
                        )

                        # suff = normalized_sufficiency_(
                        #     model = model, 
                        #     original_sentences = original_sentences, 
                        #     rationale_mask = rationale_mask, 
                        #     inputs = batch, 
                        #     full_text_probs = full_text_probs, 
                        #     full_text_class = full_text_class, 
                        #     rows = rows,
                        #     suff_y_zero = suff_y_zero,
                        #     only_query_mask=only_query_mask
                        # )
                        
                        for j_, annot_id in enumerate(batch["annotation_id"]):
                            
                            if feat_attribution_name == "--var-feat":

                                dic_name = feat_attribution_name

                            else:

                                dic_name = var_alias + "-" + feat_attribution_name

                            faithfulness_scores[annot_id][dic_name] = {
                                "comprehensiveness" : float(comp[j_]),
                                "sufficiency" : float(suff[j_]),
                                "masked prediction probs" : reduced_probs[j_].astype(np.float64).tolist(),
                                "full text prediction probs" : original_prediction[j_].astype(np.float64).tolist(),
                                "labels" : int(batch["labels"][j_].detach().cpu().item())
                            }

                    
        pbar.update(data.batch_size)


    ## save our results
    fname = args["evaluation_dir"] + args.thresholder + f"-{split}-faithfulness-metrics.json"

    with open(fname, 'w') as file:
            json.dump(
                faithfulness_scores,
                file,
                indent = 4
            ) 
    print(' save faithful metrics json file at: ', fname)
    averages = {}
    f1s_model = {
        "f1 macro avg - model labels" : {},
        "f1 macro avg - actual labels" : {}
    }

    random_annot_id = batch["annotation_id"][0]

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
    fname = args["evaluation_dir"] + args.thresholder + f"-{split}-faithfulness-metrics-description.json"

    with open(fname, 'w') as file:
            json.dump(
                averages,
                file,
                indent = 4
            ) 

    ## save f1s
    fname = args["evaluation_dir"] + args.thresholder + f"-{split}-f1-metrics-description.json"


    with open(fname, 'w') as file:
            json.dump(
                f1s_model,
                file,
                indent = 4
            ) 

    return
