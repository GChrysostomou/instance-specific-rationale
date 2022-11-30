import torch
import torch.nn as nn
import math 
import json
from tqdm import trange
import numpy as np
import os
import glob

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

from src.common_code.useful_functions import batch_from_dict_, create_only_query_mask_, create_rationale_mask_ # batch_from_dict --> batch_from_dict_
from src.common_code.metrics import normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_, normalized_comprehensiveness_soft_, normalized_sufficiency_soft_
from src.common_code.metrics import normalized_comprehensiveness_soft_2, normalized_sufficiency_soft_2
from sklearn.metrics import classification_report


def conduct_tests_(model, data, model_random_seed):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()  ### change it back later fname


## retrieve original prediction probability
    fname2 = os.path.join(
        os.getcwd(),
        args["model_dir"],
    )
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    faithfulness_results = {}
    desired_rationale_length = args.rationale_length

    # print(f"*** desired_rationale_length --> {desired_rationale_length}")

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
                "retain_gradient" : False,
            }
        print(' ------------------- ')
        #print(batch["input_ids"])
        print(batch["special_tokens"])

        # quit()
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
   
        original_prediction =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = original_prediction_output, 
                    target_key = "predicted",
                )  # return torch.tensor(new_tensor).to(device)

        ## setting up the placeholder for storing the results
        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        ## prepping for our experiments
        

        original_sentences = batch["input_ids"].clone().detach()
        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))
        
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

        ## AOPC scores and other metrics
        rationale_ratios = [0.02, 0.1, 0.2, 0.5]

        for rationale_type in {args.thresholder}:

            for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            
            for feat_name in {"random", "attention", "scaled attention", "gradients", "ig", 
            "deeplift"}: #"ig" ,"lime", "deeplift", "gradientshap",

                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

                for _i_, rationale_length in enumerate(rationale_ratios):
                    
                    ## if we are masking for a query that means we are preserving
                    ## the query and we DO NOT mask it
                    if args.query:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = rationale_type,
                            #batch_input_ids = original_sentences,
                            #special_tokens = batch["special_tokens"],
                        )

                    else:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = rationale_type
                        )

                    ## measuring faithfulness
                    comp, comp_probs  = normalized_comprehensiveness_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                    )

                    suff, suff_probs = normalized_sufficiency_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        only_query_mask=only_query_mask,
                    )


                    suff_aopc[:,_i_] = suff
                    comp_aopc[:,_i_] = comp
                    
                    ## store the ones for the desired rationale length
                    ## the rest are just for aopc
                    if rationale_length == desired_rationale_length:

                        sufficiency = suff
                        comprehensiveness = comp
                        comp_probs_save = comp_probs
                        suff_probs_save = suff_probs
                    
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency @ {desired_rationale_length}" : sufficiency[_j_],
                        f"comprehensiveness @ {desired_rationale_length}" : comprehensiveness[_j_],
                        f"masked R probs (comp) @ {desired_rationale_length}" : comp_probs_save[_j_].astype(np.float64),
                        f"only R probs (suff) @ {desired_rationale_length}" : suff_probs_save[_j_].astype(np.float64),
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
           
        pbar.update(data.batch_size)

            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", 
            "deeplift"}: #"ig", "gradientshap", , "lime"
        
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ {desired_rationale_length}"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ {desired_rationale_length}"] for k in faithfulness_results.keys()])
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    ## save all info
    fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-detailed.npy"
    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-average-description.json"

    with open(fname, 'w') as file:
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return




def conduct_experiments_zeroout_(model, data, model_random_seed,faithful_method,
                                 set, use_topk):
    ## ## retrieve importance scores
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()

    ## retrieve original prediction probability
    fname2 = os.path.join(
        os.getcwd(),
        args["model_dir"],
    )
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()


    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    faithfulness_results = {}
    desired_rationale_length = args.rationale_length

    print(f"*** desired_rationale_length --> {desired_rationale_length}")

    for batch in data:
        
        model.eval()
        model.zero_grad()

        batch = {"annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "query_mask" : batch["query_mask"].squeeze(1).to(device),
                "special_tokens" : batch["special tokens"],
                "retain_gradient" : False,
                "faithful_method": faithful_method,
                "importance_scores":torch.ones(batch["input_ids"].squeeze(1).size())
            }

        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
        original_prediction =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = original_prediction_output, 
                    target_key = "predicted",
                )  # return torch.tensor(new_tensor).to(device)

        ## setting up the placeholder for storing the results
        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        ## prepping for our experiments
        
        original_sentences = batch["input_ids"].clone().detach()
        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        original_sentences = batch["input_ids"].clone()

        rows = np.arange(batch["input_ids"].size(0))
        
        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:

            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"]
            )
            batch["input_ids"] = only_query_mask * original_sentences

        else:
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            # print('only_query_mask', only_query_mask)
            batch["input_ids"] = only_query_mask

        
        yhat, _  = model(**batch) # 此时 input id 全为o, 做的baseline ---> suff(x, y', 0)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        if use_topk:
            ###### NO NEED TO DO DIFFERENT LENGTH !!!!!!
            rationale_ratios = [0.02, 0.1, 0.2, 0.5]
            for _i_, rationale_length in enumerate(rationale_ratios):   
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                    faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                
                for feat_name in {"random", "attention", "scaled attention", "gradients", "ig", # "gradientshap",
                "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 
                    feat_score =  batch_from_dict_(
                        batch_data = batch, 
                        metadata = importance_scores, 
                        target_key = feat_name,
                    )

                    suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                    comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

                    if args.query:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #method = rationale_type,
                            batch_input_ids = original_sentences,
                            special_tokens = batch["special_tokens"],
                        )

                    else:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #method = rationale_type
                        )

                    
                    soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        importance_scores = feat_score,
                        use_topk=use_topk,
                    
                    )

                    soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        importance_scores = feat_score,
                        use_topk=use_topk,
                        only_query_mask=only_query_mask,
                    )


                    suff_aopc[:,_i_] = soft_suff
                    comp_aopc[:,_i_] = soft_comp
                    
                    ## store the ones for the desired rationale length
                    ## the rest are just for aopc
                if rationale_length == desired_rationale_length:

                        sufficiency = soft_suff
                        comprehensiveness = soft_comp
                        comp_probs_save = soft_comp_probs
                        suff_probs_save = soft_suff_probs


                    # suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
                    # comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

            for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency @ {desired_rationale_length}" : sufficiency[_j_],
                        f"comprehensiveness @ {desired_rationale_length}" : comprehensiveness[_j_],
                        f"masked R probs (comp) @ {desired_rationale_length}" : comp_probs_save[_j_].astype(np.float64),
                        f"only R probs (suff) @ {desired_rationale_length}" : suff_probs_save[_j_].astype(np.float64),
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
                


        else:
            for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            
            for feat_name in {"random", "attention", "scaled attention", "gradients", "ig", # "gradientshap",
            "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 

                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

                soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = None, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    suff_y_zero = suff_y_zero,
                    importance_scores = feat_score,
                    use_topk=use_topk,
                )

                soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = None, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    suff_y_zero = suff_y_zero,
                    importance_scores = feat_score,
                    use_topk=use_topk,
                    only_query_mask=only_query_mask,
                )

                sufficiency = soft_suff
                comprehensiveness = soft_comp
                comp_probs_save = soft_comp_probs
                suff_probs_save = soft_suff_probs


                    
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency" : soft_suff[_j_],
                        f"comprehensiveness" : soft_comp[_j_],
                        f"masked R probs (comp)" : soft_comp_probs[_j_].astype(np.float64),
                        f"only R probs (suff)" : soft_suff_probs[_j_].astype(np.float64),
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_],
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_],
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
            
        pbar.update(data.batch_size)
            
    descriptor = {}

    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",
        
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    if use_topk: 
        fname_detailed = args["evaluation_dir"] + f"ZEROOUT2-faithfulness-scores-detailed-std.npy"
        fname_descriptors = args["evaluation_dir"] + f"ZEROOUT2-faithfulness-scores-description-std.json"
    else:
                ## save all info
        fname_detailed = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-detailed-std.npy"
        ## save descriptors
        fname_descriptors = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-description-std.json"
    
    np.save(fname_detailed, faithfulness_results)
    with open(fname_descriptors, 'w') as file:
            print('  ------------->>>> saved as ', fname_descriptors)
            json.dump(
                descriptor,
                file,
                indent = 4
            )

    return



def conduct_experiments_noise_(model, data, model_random_seed,faithful_method, std, use_topk):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    ## retrieve importance scores
    importance_scores = np.load(fname, allow_pickle = True).item()

    ## retrieve original prediction probability
    fname2 = os.path.join(
        os.getcwd(),
        args["model_dir"],
    )
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()
    # 'test_1417': {'predicted': array([ 1.7342604, -1.8030814], dtype=float32), 'actual': 0}},

    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    faithfulness_results = {}
    desired_rationale_length = args.rationale_length

    print(f"*** desired_rationale_length --> {desired_rationale_length}")

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
                "retain_gradient" : False,
                "importance_scores":torch.zeros(batch["input_ids"].squeeze(1).size()),  # baseline, so all important 
                # "importance_scores": feat_scores,
                "faithful_method": faithful_method,
                "std": std,
                "add_noise": False,
            }


        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
        original_prediction =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = original_prediction_output, 
                    target_key = "predicted",
                )  # return torch.tensor(new_tensor).to(device)

        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        original_sentences = batch["input_ids"].clone().detach()
        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)
        # (batch size, class)

        full_text_probs = original_prediction.max(-1) 
        full_text_class = original_prediction.argmax(-1)

        original_sentences = batch["input_ids"].clone()

        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))

        
        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:

            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"],
            )
            batch["input_ids"] = only_query_mask * original_sentences
        else:
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            batch["input_ids"] = only_query_mask 

        
        batch["add_noise"] = True # all zero sequence and no noise!!! 
        yhat, _  = model(**batch)  # 此时 input id 完全为0
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        ) # Suff(x, ˆ y, 0) , no rationales to compare


        if use_topk:
            #print(' --------- use topk   !! for noise')
            rationale_ratios = [0.02, 0.1, 0.2, 0.5]
            for _i_, rationale_length in enumerate(rationale_ratios):          
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                    faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                
                for feat_name in {"random", "attention", "gradients", "scaled attention", "ig",# "gradientshap",
                "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 
                    feat_score =  batch_from_dict_(
                        batch_data = batch, 
                        metadata = importance_scores, 
                        target_key = feat_name,
                    )

                    suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                    comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

                    if args.query:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #method = rationale_type,
                            batch_input_ids = original_sentences,
                            special_tokens = batch["special_tokens"],
                        )
                    else:
                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #method = rationale_type
                        )

                    # 在这里面决定用 comprehensive 还是 sufficiency
                    # 在里面决定要不要加noise, 此处加,前面zeroout的baseline不加
                    soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs,   
                        full_text_class = full_text_class,  
                        rows = rows,    
                        importance_scores = feat_score,
                        suff_y_zero = suff_y_zero,
                        use_topk=use_topk,
                    )
                    soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        importance_scores = feat_score,
                        only_query_mask = only_query_mask,
                        use_topk=use_topk,
                    )
 
                    
                    suff_aopc[:,_i_] = soft_suff
                    comp_aopc[:,_i_] = soft_comp
                    
                    ## store the ones for the desired rationale length
                    ## the rest are just for aopc
                if rationale_length == desired_rationale_length:

                        sufficiency = soft_suff
                        comprehensiveness = soft_comp
                        comp_probs_save = soft_comp_probs
                        suff_probs_save = soft_suff_probs
                    
            for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency @ {desired_rationale_length}" : sufficiency[_j_],
                        f"comprehensiveness @ {desired_rationale_length}" : comprehensiveness[_j_],
                        f"masked R probs (comp) @ {desired_rationale_length}" : comp_probs_save[_j_].astype(np.float64),
                        f"only R probs (suff) @ {desired_rationale_length}" : suff_probs_save[_j_].astype(np.float64),
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
               

           
        else: # not use topk

            for _j_, annot_id in enumerate(batch["annotation_id"]):
                faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            
            for feat_name in {"random", "attention", "gradients", "scaled attention", "ig",# "gradientshap",
            "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 
                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

                # 在这里面决定用 comprehensive 还是 sufficiency
                # 在里面决定要不要加noise, 此处加,前面zeroout的baseline不加
                soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = None, 
                    inputs = batch, 
                    full_text_probs = full_text_probs,   
                    full_text_class = full_text_class,  
                    rows = rows,    
                    importance_scores = feat_score,
                    suff_y_zero = suff_y_zero,
                    use_topk=use_topk,

                )
                soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    only_query_mask=None,
                    rationale_mask = None, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    suff_y_zero = suff_y_zero,
                    importance_scores = feat_score,
                    use_topk=use_topk,
                )
                                    ## the rest are just for aopc

                # print('=====suff_aopc')
                # print(suff_aopc)
                # print('=====soft_suff')
                # print(soft_suff)

                # suff_aopc[:,:] = soft_suff
                # comp_aopc[:,:] = soft_comp

                #if rationale_length == desired_rationale_length:

                sufficiency = soft_suff
                comprehensiveness = soft_comp
                comp_probs_save = soft_comp_probs
                suff_probs_save = soft_suff_probs

                    
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency" : soft_suff[_j_],
                        f"comprehensiveness" : soft_comp[_j_],
                        f"masked R probs (comp)" : soft_comp_probs[_j_].astype(np.float64),
                        f"only R probs (suff)" : soft_suff_probs[_j_].astype(np.float64),
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_],
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_],
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
            
            
     
        pbar.update(data.batch_size)
            
            
    descriptor = {}

    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"ig", "lime","deepliftshap",
        print(faithfulness_results.keys())
        print(faithfulness_results)
        
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    


    if use_topk: 
        fname_detailed = args["evaluation_dir"] + f"NOISElimit-faithfulness-scores-detailed-std_" + str(std) + ".npy"
        fname_descriptors = args["evaluation_dir"] + f"NOISElimit-faithfulness-scores-description-std_" + str(std) + ".json"
    else:
                ## save all info
        fname_detailed = args["evaluation_dir"] + f"NOISE-faithfulness-scores-detailed-std_" + str(std) + ".npy"
        ## save descriptors
        fname_descriptors = args["evaluation_dir"] + f"NOISE-faithfulness-scores-description-std_" + str(std) + ".json"
    
    np.save(fname_detailed, faithfulness_results)
    with open(fname_descriptors, 'w') as file:
            print('  ------------->>>> saved as ', fname_descriptors)
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return



def conduct_experiments_attention_(model, data, model_random_seed, std): #faithful_method

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()



    ## retrieve original prediction probability
    fname2 = os.path.join(
        os.getcwd(),
        args["model_dir"],
    )
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()
    # 'test_1417': {'predicted': array([ 1.7342604, -1.8030814], dtype=float32), 'actual': 0}},

    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    faithfulness_results = {}
    desired_rationale_length = args.rationale_length

    print(f"*** desired_rationale_length --> {desired_rationale_length}")
    # print(importance_scores.keys())

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
                "retain_gradient" : False,
                "importance_scores":torch.zeros(batch["input_ids"].squeeze(1).size()),  # baseline, so all not important 
                "std": std,
                "add_noise": False,
            }

        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
        original_prediction =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = original_prediction_output, 
                    target_key = "predicted",
                )  # return torch.tensor(new_tensor).to(device)

        ## setting up the placeholder for storing the results
        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        original_sentences = batch["input_ids"].clone().detach()
        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)
        # (batch size, class)

        full_text_probs = original_prediction.max(-1) 
        full_text_class = original_prediction.argmax(-1)


        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))

        
        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:

            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"],
            )
            batch["input_ids"] = only_query_mask * original_sentences
        else:
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            batch["input_ids"] = only_query_mask 

        batch["add_noise"] = False
        batch["faithful_method"] = None
        yhat, _  = model(**batch)  # 此时 input id 完全为0, importance score all zero # but still got process to the perturbation model
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        ## prepping for our experiments
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        ) # Suff(x, ˆ y, 0) , no rationales to compare



        for _j_, annot_id in enumerate(batch["annotation_id"]):

                
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
        
        for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", #"gradientshap",
        "deeplift"}: # ,"lime", , "deepliftshap", 

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )


            # if args.query:

            #     rationale_mask = create_rationale_mask_(
            #         importance_scores = feat_score, 
            #         no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
            #         #method = rationale_type,
            #         batch_input_ids = original_sentences,
            #         #special_tokens = batch["special_tokens"],
            #     )

            # else:

            #     rationale_mask = create_rationale_mask_(
            #         importance_scores = feat_score, 
            #         no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
            #         #method = rationale_type
            #     )
            #  feat_score is get by new_tensor.append(metadata[_id_][target_key])

            suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
    
            # 在这里面决定用 comprehensive 还是 sufficiency
            soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                model = model, 
                original_sentences = original_sentences, 
                #rationale_mask = rationale_mask,
                inputs = batch, 
                full_text_probs = full_text_probs,   
                full_text_class = full_text_class,  
                rows = rows,    
                importance_scores = feat_score,
                suff_y_zero = suff_y_zero,
            )
            soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                model = model, 
                original_sentences = original_sentences, 
                #rationale_mask = rationale_mask, 
                inputs = batch, 
                full_text_probs = full_text_probs, 
                full_text_class = full_text_class, 
                rows = rows,
                suff_y_zero = suff_y_zero,
                importance_scores = feat_score,
                #only_query_mask=only_query_mask,
            )

                
            for _j_, annot_id in enumerate(batch["annotation_id"]):
                
                faithfulness_results[annot_id][feat_name] = {
                    f"sufficiency" : soft_suff[_j_],
                    f"comprehensiveness" : soft_comp[_j_],
                    f"masked R probs (comp)" : soft_comp_probs[_j_].astype(np.float64),
                    f"only R probs (suff)" : soft_suff_probs[_j_].astype(np.float64),
                    "sufficiency aopc" : {
                        "mean" : suff_aopc[_j_],
                        "per ratio" : suff_aopc[_j_]
                    },
                    "comprehensiveness aopc" : {
                        "mean" : comp_aopc[_j_],
                        "per ratio" : comp_aopc[_j_]
                    }
                }
        
           
        pbar.update(data.batch_size)

            
    descriptor = {}

    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",
        
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    ## save all info
    fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-detailed.npy"

    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-description.json"

    with open(fname, 'w') as file:
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return


def conduct_experiments_attention_2(model, data, model_random_seed, std):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()  ### change it back later fname


## retrieve original prediction probability
    fname2 = os.path.join(
        os.getcwd(),
        args["model_dir"],
    )
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    faithfulness_results = {}
    desired_rationale_length = args.rationale_length

    # print(f"*** desired_rationale_length --> {desired_rationale_length}")

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
                "retain_gradient" : False,
                "importance_scores":torch.zeros(batch["input_ids"].squeeze(1).size()),  # baseline, so all not important 
                "std": std,
                "add_noise": False,
            }
            
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
   
        original_prediction =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = original_prediction_output, 
                    target_key = "predicted",
                )  # return torch.tensor(new_tensor).to(device)

        ## setting up the placeholder for storing the results
        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        ## prepping for our experiments
        original_sentences = batch["input_ids"].clone().detach()
        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        rows = np.arange(batch["input_ids"].size(0))
        
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


        batch["add_noise"] = False
        batch["faithful_method"] = None
        yhat, _  = model(**batch)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        ## AOPC scores and other metrics
        rationale_ratios = [0.02, 0.1, 0.2, 0.5]

        for _j_, annot_id in enumerate(batch["annotation_id"]):
                
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
        
        for feat_name in {"random", "attention", "scaled attention", "gradients", "ig", 
        "deeplift"}: #"ig" ,"lime", "deeplift", "gradientshap",

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

            for _i_, rationale_length in enumerate(rationale_ratios):
                
                ## if we are masking for a query that means we are preserving
                ## the query and we DO NOT mask it
                if args.query:

                    rationale_mask = create_rationale_mask_(
                        importance_scores = feat_score, 
                        no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                        method = 'topk',
                        batch_input_ids = original_sentences,
                        special_tokens = batch["special_tokens"],
                    )

                else:

                    rationale_mask = create_rationale_mask_(
                        importance_scores = feat_score, 
                        no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                        method = 'topk',
                    )

                ## measuring faithfulness
                comp, comp_probs  = normalized_comprehensiveness_soft_2(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = rationale_mask, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    importance_scores = feat_score,
                    suff_y_zero = suff_y_zero,
                )

                suff, suff_probs = normalized_sufficiency_soft_2(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = rationale_mask, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    suff_y_zero = suff_y_zero,
                    importance_scores = feat_score,
                    only_query_mask=only_query_mask,
                )


                suff_aopc[:,_i_] = suff
                comp_aopc[:,_i_] = comp
                
                ## store the ones for the desired rationale length
                ## the rest are just for aopc
            if rationale_length == desired_rationale_length:

                    sufficiency = suff
                    comprehensiveness = comp
                    comp_probs_save = comp_probs
                    suff_probs_save = suff_probs
                
        for _j_, annot_id in enumerate(batch["annotation_id"]):
                
                faithfulness_results[annot_id][feat_name] = {
                    f"sufficiency @ {desired_rationale_length}" : sufficiency[_j_],
                    f"comprehensiveness @ {desired_rationale_length}" : comprehensiveness[_j_],
                    f"masked R probs (comp) @ {desired_rationale_length}" : comp_probs_save[_j_].astype(np.float64),
                    f"only R probs (suff) @ {desired_rationale_length}" : suff_probs_save[_j_].astype(np.float64),
                    "sufficiency aopc" : {
                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                        "per ratio" : suff_aopc[_j_]
                    },
                    "comprehensiveness aopc" : {
                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                        "per ratio" : comp_aopc[_j_]
                    }
                }
        
    
        
        
        pbar.update(data.batch_size)

            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", 
            "deeplift"}: #"ig", "gradientshap", , "lime"
        
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ {desired_rationale_length}"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ {desired_rationale_length}"] for k in faithfulness_results.keys()])
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    ## save all info
    fname = args["evaluation_dir"] + f"ATTENTION2-faithfulness-scores-detailed.npy"

    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = args["evaluation_dir"] + f"ATTENTION2-faithfulness-scores-description.json"


    with open(fname, 'w') as file:
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return

