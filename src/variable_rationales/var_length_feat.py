import torch
from torch import nn
import math 
import json
import numpy as np
from src.common_code.metrics import jsd, kl_div_loss, perplexity, simple_diff
import time

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from src.common_code.useful_functions import mask_topk, mask_contigious, batch_from_dict, create_rationale_mask_, create_only_query_mask_
import math

nn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)


div_funs = {
    "jsd":jsd, 
    "perplexity":perplexity, 
    "kldiv":kl_div_loss,
    "classdiff": simple_diff
}


def rationale_length_computer_(
    model, inputs, scores, y_original, 
    results_dict, feature_attribution, 
    zero_logits, original_sents, 
    fidelity = "lower_fidelity"):


    divergence_fun = div_funs[args.divergence]

    """
    function to calculate for a batch:
        * variable rationale length
        * variable rationale mask
        * fixed rationale mask
    for a specific set of importance scores
    """

    assert fidelity in ["max_fidelity", "lower_fidelity"]

    tokens = args.rationale_length  * (inputs["lengths"] - 2).float().mean()

    tokens = int(max(1, torch.round(tokens)))
    
    ## if we break down our search in increments
    if fidelity == "lower_fidelity":
        
        per_how_many = 5/100 ## skip every per_how_many percent of tokens
        percent_to_tokens = round(per_how_many * int(min(inputs["lengths"]))) ## translate percentage to tokens

        ## special case for very short sequences in SST and AG of less than 6 tokens
        if percent_to_tokens == 0:
            
            collector = torch.zeros([tokens, original_sents.size(0)])

            grange = range(1, tokens + 1)
        
        ## for longer than 4 word sequences
        else:

            grange = range(percent_to_tokens, tokens + percent_to_tokens, percent_to_tokens) ## convert to range with increments

            ## empty matrix to collect scores 
            ## // +1 to keep empty first column like below (0 token)
            collector = torch.zeros([len(grange), original_sents.size(0)])
        
    ## else if we consider and search on every token
    else:
        
        collector = torch.zeros([tokens, original_sents.size(0)])

        grange = range(1, tokens + 1)

    model.eval()
    stepwise_preds = []
    ## begin search
    start_time = time.time()

    token_collector = []
    with torch.no_grad():
        
        for j, _tok in enumerate(grange):
            
            ## min 1
            if _tok == 0: _tok = 1
            ## ensure we do not go over
            if _tok > tokens: _tok = tokens

            rationale_mask = create_rationale_mask_(
                importance_scores = scores, 
                no_of_masked_tokens = np.array([_tok]*scores.size(0)),
                method = args.thresholder
            )

            inputs["input_ids"] = (rationale_mask == 0).long() * original_sents
    
            yhat, _ = model(**inputs)

            stepwise_preds.append(yhat.argmax(-1).detach().cpu().numpy())

            full_div = divergence_fun(
                torch.softmax(y_original, dim = -1), 
                torch.softmax(yhat, dim = -1)
            ) 

            collector[j] = full_div.detach().cpu()

            token_collector.append(_tok)

    #### in short sequences (e.g. where grange is 0) it means they are formed from one token
    #### so that token is our explanation

    stepwise_preds = np.stack(stepwise_preds).T

    assert stepwise_preds.shape[0] == y_original.size(0)

    max_div, indxes = collector.max(0)

    end_time = time.time()

    ## now to generate the rationale ratio
    ## and other data that we care about saving
    for _i_ in range(y_original.size(0)):

        annot_id = inputs["annotation_id"][_i_]
        fixed_rationale_length = math.ceil(args.rationale_length * inputs["lengths"][_i_].float())
        full_text_length = inputs["lengths"][_i_]
        rationale_length = token_collector[indxes[_i_].detach().cpu().item()]
        rationale_ratio = rationale_length / (full_text_length.float().detach().cpu().item()-1)
        
        ## now to create the mask of variable rationales
        ## rationale selected (with 1's)  
        if args.thresholder == "topk":

            rationale_mask = (mask_topk(original_sents[_i_], scores[_i_], rationale_length) == 0).long().detach().cpu().numpy()

        else:

            rationale_mask = (mask_contigious(original_sents[_i_], scores[_i_],rationale_length) == 0).long().detach().cpu().numpy()
         
        ## now to create the mask of variable rationales
        ## rationale selected (with 1's)
        if args.thresholder == "topk":

            fixed_rationale_mask = (mask_topk(original_sents[_i_], scores[_i_], fixed_rationale_length) == 0).long().detach().cpu().numpy()

        else:

            fixed_rationale_mask = (mask_contigious(original_sents[_i_], scores[_i_],fixed_rationale_length) == 0).long().detach().cpu().numpy()

        ## we also need the fixed max div for selecting through the best rationales
        fixed_div = collector[-1,:][_i_] ## the last row of divergences represents the max rationale length

        results_dict[annot_id][feature_attribution] = {
            "variable rationale length" : rationale_length,
            "fixed rationale length" : fixed_rationale_length,
            "variable rationale ratio" : rationale_ratio,
            "variable rationale mask" : rationale_mask,
            "fixed rationale mask" : fixed_rationale_mask,
            f"fixed-length divergence" : fixed_div.cpu().item(),
            f"variable-length divergence" : max_div[_i_].cpu().item(),
            "running predictions" : stepwise_preds[_i_],
            "time elapsed" : end_time - start_time
        }

    return 

import os 
from tqdm import trange

def get_rationale_metadata_(model, data_split_name, data, model_random_seed):

    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        data_split_name + "-rationale_metadata.npy"
    )

    if os.path.isfile(fname):

        print(f"rationale metadata file exists at {fname}") 
        print("remove if you would like to rerun")

        return 

    desc = f'creating rationale data for -> {data_split_name}'

    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    fname += f"{data_split_name}_importance_scores_{model_random_seed}.npy"

    ## retrieve importance scores
    importance_scores = np.load(fname, allow_pickle = True).item()

    
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    
    rationale_results = {}

    for batch in data:
        
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
                "retain_gradient" : True
            }
            
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
   
        original_prediction, _ =  model(**batch)

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        for _i_ in range(original_prediction.size(0)):
            
            annotation_id = batch["annotation_id"][_i_]
            
            ## setting up the placeholder for storing the  rationales
            rationale_results[annotation_id] = {}
            rationale_results[annotation_id]["original prediction"] = original_prediction[_i_].detach().cpu().numpy()
            rationale_results[annotation_id]["thresholder"] = args.thresholder
            rationale_results[annotation_id]["divergence metric"] = args.divergence

        original_sents = batch["input_ids"].clone()

        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:

            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"]
            )

            batch["input_ids"] = only_query_mask * original_sents

        else:

            only_query_mask=torch.zeros_like(batch["input_ids"]).long()

            batch["input_ids"] = only_query_mask

        zero_logits, _ =  model(**batch)

        
        ## percentage of flips
        for feat_name in {"lime", "random", "attention",  "gradients",   "ig", "scaled attention", "deeplift"}:
            
            feat_score = batch_from_dict(
                batch_data = batch,
                metadata = importance_scores,
                target_key =  feat_name,
                extra_layer = None
            )

            rationale_length_computer_(
                model = model, 
                inputs = batch, 
                scores = feat_score, 
                y_original = original_prediction, 
                zero_logits = zero_logits,
                original_sents=original_sents,
                fidelity = "lower_fidelity",
                feature_attribution = feat_name, 
                results_dict = rationale_results
            )

        ## select best fixed (fixed-len + var-feat) and variable rationales (var-len + var-feat) and save 
        for _i_ in range(original_sents.size(0)):

            annotation_id = batch["annotation_id"][_i_]

            ## initiators
            init_fixed_div = float("-inf")
            init_var_div = float("-inf")

            for feat_name in {"attention", "scaled attention", "gradients", "ig", "lime", "deeplift"}:
                
                fixed_div = rationale_results[annotation_id][feat_name]["fixed-length divergence"]
                var_div = rationale_results[annotation_id][feat_name]["variable-length divergence"]

                if fixed_div > init_fixed_div:

                    rationale_results[annotation_id]["fixed-len_var-feat"] = rationale_results[annotation_id][feat_name]
                    rationale_results[annotation_id]["fixed-len_var-feat"]["feature attribution name"] = feat_name

                    init_fixed_div = fixed_div


                if var_div > init_var_div:

                    rationale_results[annotation_id]["var-len_var-feat"] = rationale_results[annotation_id][feat_name]
                    rationale_results[annotation_id]["var-len_var-feat"]["feature attribution name"] = feat_name

                    init_var_div = var_div

        pbar.update(data.batch_size)

    ## save rationale masks
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        ""
    )

    os.makedirs(fname, exist_ok= True)

    print(f"saved -> {fname}")

    np.save(fname + data_split_name + "-rationale_metadata.npy", rationale_results)

    return

            