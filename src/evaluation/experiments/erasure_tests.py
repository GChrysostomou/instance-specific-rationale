from pickle import NONE
from re import T
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
from src.common_code.metrics import comprehensiveness_, normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_, normalized_comprehensiveness_soft_, normalized_sufficiency_soft_
from sklearn.metrics import classification_report


feat_name_dict = {"attention", "scaled attention", "gradients", "ig", "deeplift", "random"} #"gradientshap"
#feat_name_dict = {"deeplift"} #"gradientshap"
rationale_ratios = [1.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]   # [1.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 



def conduct_tests_(model, data, model_random_seed):    
    rationale_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
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

    for i, batch in enumerate(data):

        #print(' len =======', batch['lengths'])
        
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

        ####################### check baseline y suff zero
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
        ####################### DONE check baseline y suff zero

        for _j_, annot_id in enumerate(batch["annotation_id"]):
                
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            for feat in feat_name_dict:
                faithfulness_results[annot_id][feat] = {}
                
        for feat_name in feat_name_dict:#feat_name_dict: #"ig" ,"lime", "deeplift", "gradientshap",
            # print('  ', feat_name)
            # print('  ', feat_name)
            # print('  ', feat_name)
            # print('  ', feat_name)

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)


            for _i_, rationale_length in enumerate(rationale_ratios):
                # print(' ')
                # print('  ------- rationale_length', rationale_length)

                if args.query:

                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #no_of_masked_tokens = tempC,
                            method = 'topk',
                            batch_input_ids = original_sentences,
                            special_tokens = batch["special_tokens"],
                        )

                else:
                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = 'topk',
                            special_tokens = batch["special_tokens"],
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



                comp, comp_probs  = normalized_comprehensiveness_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = rationale_mask, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    #suff_y_zero = suff_y_zero,
                    suff_y_zero = suff_y_zero,
                )

                # print(' ------------  SUFF  ----------')
                # print(suff, suff_probs)
                # print(' ------------  COMP  ----------')
                # print(comp, comp_probs)


           
                suff_aopc[:,_i_] = suff
                comp_aopc[:,_i_] = comp
                
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                        # faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                        # faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                    
                    faithfulness_results[annot_id][feat_name][f"sufficiency @ {rationale_length}"] = suff[_j_]
                    faithfulness_results[annot_id][feat_name][f"comprehensiveness @ {rationale_length}"] = comp[_j_]
                    faithfulness_results[annot_id][feat_name][f"masked R probs (comp) @ {rationale_length}"] = comp_probs[_j_].astype(np.float64)
                    faithfulness_results[annot_id][feat_name][f"only R probs (suff) @ {rationale_length}"] = suff_probs[_j_].astype(np.float64)
                

                    
                    if _i_ == len(rationale_ratios)-1:
                        faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : suff_aopc[_j_]
                                                                        }
                        faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : comp_aopc[_j_]
                                                                        }
        
            #quit()
        pbar.update(data.batch_size)


            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", 
            "deeplift"}: #"ig", "gradientshap", , "lime"

        
        sufficiencies_001 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.01"] for k in faithfulness_results.keys()])
        comprehensivenesses_001 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.01"] for k in faithfulness_results.keys()])

        sufficiencies_002 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.02"] for k in faithfulness_results.keys()])
        comprehensivenesses_002 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.02"] for k in faithfulness_results.keys()])

        sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.05"] for k in faithfulness_results.keys()])
        comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

        sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.1"] for k in faithfulness_results.keys()])
        comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

        sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.2"] for k in faithfulness_results.keys()])
        comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

        sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
        comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

        sufficiencies_10 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 1.0"] for k in faithfulness_results.keys()])
        comprehensivenesses_10 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 1.0"] for k in faithfulness_results.keys()])

        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        
        descriptor[feat_attr] = {
            "sufficiencies @ 0.01" : {
                "mean" : sufficiencies_001.mean(),
                "std" : sufficiencies_001.std()
            },
            "comprehensiveness @ 0.01" : {
                "mean" : comprehensivenesses_001.mean(),
                "std" : comprehensivenesses_001.std()
            },

            "sufficiencies @ 0.02" : {
                "mean" : sufficiencies_002.mean(),
                "std" : sufficiencies_002.std()
            },
            "comprehensiveness @ 0.02" : {
                "mean" : comprehensivenesses_002.mean(),
                "std" : comprehensivenesses_002.std()
            },

            "sufficiencies @ 0.05" : {
                "mean" : sufficiencies_005.mean(),
                "std" : sufficiencies_005.std()
            },
            "comprehensiveness @ 0.05" : {
                "mean" : comprehensivenesses_005.mean(),
                "std" : comprehensivenesses_005.std()
            },


            "sufficiencies @ 0.1" : {
                "mean" : sufficiencies_01.mean(),
                "std" : sufficiencies_01.std()
            },
            "comprehensiveness @ 0.1" : {
                "mean" : comprehensivenesses_01.mean(),
                "std" : comprehensivenesses_01.std()
            },

            
            "sufficiencies @ 0.2" : {
                "mean" : sufficiencies_02.mean(),
                "std" : sufficiencies_02.std()
            },
            "comprehensiveness @ 0.2" : {
                "mean" : comprehensivenesses_02.mean(),
                "std" : comprehensivenesses_02.std()
            },
            

            "sufficiencies @ 0.5" : {
                "mean" : sufficiencies_05.mean(),
                "std" : sufficiencies_05.std()
            },
            "comprehensiveness @ 0.5" : {
                "mean" : comprehensivenesses_05.mean(),
                "std" : comprehensivenesses_05.std()
            },


            "sufficiencies @ 1.0" : {
                "mean" : sufficiencies_10.mean(),
                "std" : sufficiencies_10.std()
            },
            "comprehensiveness @ 1.0" : {
                "mean" : comprehensivenesses_10.mean(),
                "std" : comprehensivenesses_10.std()
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


def conduct_experiments_noise_2(model, data, model_random_seed,
                                std, use_topk,normalise):    
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

    for i, batch in enumerate(data):
        
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

        ####################### check baseline y suff zero
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
        ####################### DONE check baseline y suff zero

        for _j_, annot_id in enumerate(batch["annotation_id"]):
                
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            for feat in feat_name_dict:
                faithfulness_results[annot_id][feat] = {}
                
        for feat_name in feat_name_dict:#feat_name_dict: #"ig" ,"lime", "deeplift", "gradientshap",

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)


            for _i_, rationale_length in enumerate(rationale_ratios):
                # print(' ')
                # print('  ------- rationale_length', rationale_length)

                if args.query:

                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #no_of_masked_tokens = tempC,
                            method = 'topk',
                            batch_input_ids = original_sentences,
                            special_tokens = batch["special_tokens"],
                        )

                    

                else:
                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = 'topk',
                            special_tokens = batch["special_tokens"],
                        )
                        

            
                suff, suff_probs = normalized_sufficiency_soft_(
                
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



                comp, comp_probs  = normalized_comprehensiveness_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = rationale_mask, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    #suff_y_zero = suff_y_zero,
                    suff_y_zero = suff_y_zero,
                )

                # print(' ------------  SUFF  ----------')
                # print(suff, suff_probs)
                # print(' ------------  COMP  ----------')
                # print(comp, comp_probs)


           
                suff_aopc[:,_i_] = suff
                comp_aopc[:,_i_] = comp
                
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                        # faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                        # faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                    
                    faithfulness_results[annot_id][feat_name][f"sufficiency @ {rationale_length}"] = suff[_j_]
                    faithfulness_results[annot_id][feat_name][f"comprehensiveness @ {rationale_length}"] = comp[_j_]
                    faithfulness_results[annot_id][feat_name][f"masked R probs (comp) @ {rationale_length}"] = comp_probs[_j_].astype(np.float64)
                    faithfulness_results[annot_id][feat_name][f"only R probs (suff) @ {rationale_length}"] = suff_probs[_j_].astype(np.float64)
                

                    
                    if _i_ == len(rationale_ratios)-1:
                        faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : suff_aopc[_j_]
                                                                        }
                        faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : comp_aopc[_j_]
                                                                        }
        
            #quit()
        pbar.update(data.batch_size)


            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", 
            "deeplift"}: #"ig", "gradientshap", , "lime"

        
        sufficiencies_001 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.01"] for k in faithfulness_results.keys()])
        comprehensivenesses_001 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.01"] for k in faithfulness_results.keys()])

        sufficiencies_002 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.02"] for k in faithfulness_results.keys()])
        comprehensivenesses_002 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.02"] for k in faithfulness_results.keys()])

        sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.05"] for k in faithfulness_results.keys()])
        comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

        sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.1"] for k in faithfulness_results.keys()])
        comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

        sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.2"] for k in faithfulness_results.keys()])
        comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

        sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
        comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

        sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
        comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

        sufficiencies_10 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 1.0"] for k in faithfulness_results.keys()])
        comprehensivenesses_10 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 1.0"] for k in faithfulness_results.keys()])

        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        
        descriptor[feat_attr] = {
            "sufficiencies @ 0.01" : {
                "mean" : sufficiencies_001.mean(),
                "std" : sufficiencies_001.std()
            },
            "comprehensiveness @ 0.01" : {
                "mean" : comprehensivenesses_001.mean(),
                "std" : comprehensivenesses_001.std()
            },

            "sufficiencies @ 0.02" : {
                "mean" : sufficiencies_002.mean(),
                "std" : sufficiencies_002.std()
            },
            "comprehensiveness @ 0.02" : {
                "mean" : comprehensivenesses_002.mean(),
                "std" : comprehensivenesses_002.std()
            },

            "sufficiencies @ 0.05" : {
                "mean" : sufficiencies_005.mean(),
                "std" : sufficiencies_005.std()
            },
            "comprehensiveness @ 0.05" : {
                "mean" : comprehensivenesses_005.mean(),
                "std" : comprehensivenesses_005.std()
            },


            "sufficiencies @ 0.1" : {
                "mean" : sufficiencies_01.mean(),
                "std" : sufficiencies_01.std()
            },
            "comprehensiveness @ 0.1" : {
                "mean" : comprehensivenesses_01.mean(),
                "std" : comprehensivenesses_01.std()
            },

            
            "sufficiencies @ 0.2" : {
                "mean" : sufficiencies_02.mean(),
                "std" : sufficiencies_02.std()
            },
            "comprehensiveness @ 0.2" : {
                "mean" : comprehensivenesses_02.mean(),
                "std" : comprehensivenesses_02.std()
            },
            

            "sufficiencies @ 0.5" : {
                "mean" : sufficiencies_05.mean(),
                "std" : sufficiencies_05.std()
            },
            "comprehensiveness @ 0.5" : {
                "mean" : comprehensivenesses_05.mean(),
                "std" : comprehensivenesses_05.std()
            },


            "sufficiencies @ 1.0" : {
                "mean" : sufficiencies_10.mean(),
                "std" : sufficiencies_10.std()
            },
            "comprehensiveness @ 1.0" : {
                "mean" : comprehensivenesses_10.mean(),
                "std" : comprehensivenesses_10.std()
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

def conduct_tests_2(model, data, model_random_seed):    
    rationale_ratios = [0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
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

    for i, batch in enumerate(data):

        #print(' len =======', batch['lengths'])
        
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

        ####################### check baseline y suff zero
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
        ####################### DONE check baseline y suff zero

        for _j_, annot_id in enumerate(batch["annotation_id"]):
                
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            for feat in feat_name_dict:
                faithfulness_results[annot_id][feat] = {}
                
        for feat_name in feat_name_dict:#feat_name_dict: #"ig" ,"lime", "deeplift", "gradientshap",
            # print('  ', feat_name)
            # print('  ', feat_name)
            # print('  ', feat_name)
            # print('  ', feat_name)

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)


            for _i_, rationale_length in enumerate(rationale_ratios):
                # print(' ')
                # print('  ------- rationale_length', rationale_length)

                if args.query:

                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            #no_of_masked_tokens = tempC,
                            method = 'topk',
                            batch_input_ids = original_sentences,
                            special_tokens = batch["special_tokens"],
                        )

                else:
                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = 'topk',
                            special_tokens = batch["special_tokens"],
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



                comp, comp_probs  = normalized_comprehensiveness_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = rationale_mask, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    #suff_y_zero = suff_y_zero,
                    suff_y_zero = suff_y_zero,
                )

                # print(' ------------  SUFF  ----------')
                # print(suff, suff_probs)
                # print(' ------------  COMP  ----------')
                # print(comp, comp_probs)


           
                suff_aopc[:,_i_] = suff
                comp_aopc[:,_i_] = comp
                
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                        # faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                        # faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                    
                    faithfulness_results[annot_id][feat_name][f"sufficiency @ {rationale_length}"] = suff[_j_]
                    faithfulness_results[annot_id][feat_name][f"comprehensiveness @ {rationale_length}"] = comp[_j_]
                    faithfulness_results[annot_id][feat_name][f"masked R probs (comp) @ {rationale_length}"] = comp_probs[_j_].astype(np.float64)
                    faithfulness_results[annot_id][feat_name][f"only R probs (suff) @ {rationale_length}"] = suff_probs[_j_].astype(np.float64)
                

                    
                    if _i_ == len(rationale_ratios)-1:
                        faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : suff_aopc[_j_]
                                                                        }
                        faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : comp_aopc[_j_]
                                                                        }
        
            #quit()
        pbar.update(data.batch_size)


            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", 
            "deeplift"}: #"ig", "gradientshap", , "lime" [0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

        
        sufficiencies_03 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.3"] for k in faithfulness_results.keys()])
        comprehensivenesses_03 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.3"] for k in faithfulness_results.keys()])

        sufficiencies_04 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.4"] for k in faithfulness_results.keys()])
        comprehensivenesses_04 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.4"] for k in faithfulness_results.keys()])

        sufficiencies_06 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.6"] for k in faithfulness_results.keys()])
        comprehensivenesses_06 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.6"] for k in faithfulness_results.keys()])

        sufficiencies_07 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.7"] for k in faithfulness_results.keys()])
        comprehensivenesses_07 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.7"] for k in faithfulness_results.keys()])

        sufficiencies_08 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.8"] for k in faithfulness_results.keys()])
        comprehensivenesses_08 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.8"] for k in faithfulness_results.keys()])

        sufficiencies_09 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.9"] for k in faithfulness_results.keys()])
        comprehensivenesses_09 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.9"] for k in faithfulness_results.keys()])

        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        
        descriptor[feat_attr] = {
            "sufficiencies @ 0.3" : {
                "mean" : sufficiencies_03.mean(),
                "std" : sufficiencies_03.std()
            },
            "comprehensiveness @ 0.3" : {
                "mean" : comprehensivenesses_03.mean(),
                "std" : comprehensivenesses_03.std()
            },

            "sufficiencies @ 0.4" : {
                "mean" : sufficiencies_04.mean(),
                "std" : sufficiencies_04.std()
            },
            "comprehensiveness @ 0.4" : {
                "mean" : comprehensivenesses_04.mean(),
                "std" : comprehensivenesses_04.std()
            },

            "sufficiencies @ 0.6" : {
                "mean" : sufficiencies_06.mean(),
                "std" : sufficiencies_06.std()
            },
            "comprehensiveness @ 0.6" : {
                "mean" : comprehensivenesses_06.mean(),
                "std" : comprehensivenesses_06.std()
            },


            "sufficiencies @ 0.7" : {
                "mean" : sufficiencies_07.mean(),
                "std" : sufficiencies_07.std()
            },
            "comprehensiveness @ 0.7" : {
                "mean" : comprehensivenesses_07.mean(),
                "std" : comprehensivenesses_07.std()
            },

            
            "sufficiencies @ 0.8" : {
                "mean" : sufficiencies_08.mean(),
                "std" : sufficiencies_08.std()
            },
            "comprehensiveness @ 0.8" : {
                "mean" : comprehensivenesses_08.mean(),
                "std" : comprehensivenesses_08.std()
            },
            

            "sufficiencies @ 0.9" : {
                "mean" : sufficiencies_09.mean(),
                "std" : sufficiencies_09.std()
            },
            "comprehensiveness @ 0.9" : {
                "mean" : comprehensivenesses_09.mean(),
                "std" : comprehensivenesses_09.std()
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
    fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-detailed2.npy"
    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-average-description2.json"

    with open(fname, 'w') as file:
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return




def conduct_experiments_zeroout_(model, data, model_random_seed, use_topk, normalise): # 0 for no normalization, 1 for softmax

    fname = os.path.join(os.getcwd(),args["data_dir"], "importance_scores","" )
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()

    ## retrieve original prediction probability
    fname2 = os.path.join(os.getcwd(),args["model_dir"])
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    desired_rationale_length = args.rationale_length


    faithfulness_results = {}
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
        
        if args.query:
            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"],
            )
            batch["input_ids"] = only_query_mask * original_sentences
        else:
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            batch["input_ids"] = only_query_mask


        batch["faithful_method"] = "soft_suff"
        batch["importance_scores"]=torch.zeros(batch["input_ids"].squeeze(1).size())
        batch["rationale_mask"]=torch.zeros(batch["input_ids"].size())
        batch["add_noise"]=True  # 测试点 zerrout 和 noise 定用 true
        yhat, _  = model(**batch) # 此时 input id 全为o, 做的baseline ---> suff(x, y', 0)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )



        batch["add_noise"]=True

        for _j_, annot_id in enumerate(batch["annotation_id"]):
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            for feat in feat_name_dict:
                faithfulness_results[annot_id][feat] = {}

       

            
        for feat_name in feat_name_dict: #"ig" ,"lime", "deeplift", "deepliftshap", 

            feat_score =  batch_from_dict_(batch_data = batch, metadata = importance_scores, target_key = feat_name,)
            #print("==>> feat: ", feat_name)



            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            
            for _i_, rationale_length in enumerate(rationale_ratios):   
                
                if rationale_length == 1.0: 
                    rationale_mask= torch.ones(batch["input_ids"].size())



                    if normalise == 5:
                    
                        soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                            model2 = model, 
                            original_sentences = original_sentences, 
                            rationale_mask = rationale_mask, 
                            inputs = batch, 
                            full_text_probs = full_text_probs, 
                            full_text_class = full_text_class, 
                            rows = rows,
                            suff_y_zero = suff_y_zero,
                            importance_scores = feat_score,
                            use_topk=True,
                            normalise=normalise,
                        )
                        soft_suff = soft_comp
                        soft_suff_probs = soft_comp_probs
                    else: 
                        soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                            model2 = model, 
                            original_sentences = original_sentences, 
                            #rationale_mask = rationale_mask, 
                            inputs = batch, 
                            full_text_probs = full_text_probs, 
                            full_text_class = full_text_class, 
                            rows = rows,
                            suff_y_zero = suff_y_zero,
                            importance_scores = feat_score,
                            use_topk=True,
                            only_query_mask=only_query_mask,
                            normalise=normalise,
                        )
                        soft_comp = soft_suff
                        soft_comp_probs = soft_suff_probs

                    # quit()

                
                else:
                    pass # return

                    # if args.query:
                    #     rationale_mask = create_rationale_mask_(
                    #         importance_scores = feat_score, 
                    #         no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                    #         method = 'topk',
                    #         batch_input_ids = original_sentences,
                    #         special_tokens = batch["special_tokens"],
                    # )
                    # else:
                    #     rationale_mask = create_rationale_mask_(
                    #         importance_scores = feat_score, 
                    #         no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                    #         method = 'topk',
                    #         special_tokens = batch["special_tokens"],
                    #     )

                    # if normalise == 5:
                    
                    #     soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                    #         model = model, 
                    #         original_sentences = original_sentences, 
                    #         #rationale_mask = rationale_mask, 
                    #         inputs = batch, 
                    #         full_text_probs = full_text_probs, 
                    #         full_text_class = full_text_class, 
                    #         rows = rows,
                    #         suff_y_zero = suff_y_zero,
                    #         importance_scores = feat_score,
                    #         use_topk=True,
                    #         normalise=normalise,
                    #     )
                    #     soft_suff = soft_comp
                    #     soft_suff_probs = soft_comp_probs
                    # else: 
                    #     soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                    #         model = model, 
                    #         original_sentences = original_sentences, 
                    #         #rationale_mask = rationale_mask, 
                    #         inputs = batch, 
                    #         full_text_probs = full_text_probs, 
                    #         full_text_class = full_text_class, 
                    #         rows = rows,
                    #         suff_y_zero = suff_y_zero,
                    #         importance_scores = feat_score,
                    #         use_topk=True,
                    #         only_query_mask=only_query_mask,
                    #         normalise=normalise,
                    #     )
                    #     soft_comp = soft_suff
                    #     soft_comp_probs = soft_suff_probs



                    # quit()


                suff_aopc[:,_i_] = soft_suff  # id, lenght
                comp_aopc[:,_i_] = soft_comp
                

                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    # faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                    # faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                
                    faithfulness_results[annot_id][feat_name][f"sufficiency @ {rationale_length}"] = soft_suff[_j_]
                    faithfulness_results[annot_id][feat_name][f"comprehensiveness @ {rationale_length}"] = soft_comp[_j_]
                    faithfulness_results[annot_id][feat_name][f"masked R probs (comp) @ {rationale_length}"] = soft_comp_probs[_j_].astype(np.float64)
                    faithfulness_results[annot_id][feat_name][f"only R probs (suff) @ {rationale_length}"] = soft_suff_probs[_j_].astype(np.float64)
                

                    
                    if _i_ == len(rationale_ratios)-1:
                        faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : suff_aopc[_j_]
                                                                        }
                        faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : comp_aopc[_j_]
                                                                        }

    
    pbar.update(data.batch_size)


    detailed_fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-normal_{normalise}.npy"
        #description_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-description.json"
    np.save(detailed_fname, faithfulness_results)
            


    descriptor = {}

    # filling getting averages
    for feat_attr in feat_name_dict: #"gradientshap", "lime","deepliftshap",   [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
        
        if use_topk: # 0.05, 0.1, 0.2, 0.5]
            # sufficiencies_001 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.01"] for k in faithfulness_results.keys()])
            # comprehensivenesses_001 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.01"] for k in faithfulness_results.keys()])

            # sufficiencies_002 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.02"] for k in faithfulness_results.keys()])
            # comprehensivenesses_002 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.02"] for k in faithfulness_results.keys()])

            # sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.05"] for k in faithfulness_results.keys()])
            # comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

            # sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.1"] for k in faithfulness_results.keys()])
            # comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

            # sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.2"] for k in faithfulness_results.keys()])
            # comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

            # sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
            # comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

            # sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
            # comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

            sufficiencies_10 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 1.0"] for k in faithfulness_results.keys()])
            comprehensivenesses_10 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 1.0"] for k in faithfulness_results.keys()])

            aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
            aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])
            
            descriptor[feat_attr] = {
                # "sufficiencies @ 0.01" : {
                #     "mean" : sufficiencies_001.mean(),
                #     "std" : sufficiencies_001.std()
                # },
                # "comprehensiveness @ 0.01" : {
                #     "mean" : comprehensivenesses_001.mean(),
                #     "std" : comprehensivenesses_001.std()
                # },


                # "sufficiencies @ 0.02" : {
                #     "mean" : sufficiencies_002.mean(),
                #     "std" : sufficiencies_002.std()
                # },
                # "comprehensiveness @ 0.02" : {
                #     "mean" : comprehensivenesses_002.mean(),
                #     "std" : comprehensivenesses_002.std()
                # },

                # "sufficiencies @ 0.05" : {
                #     "mean" : sufficiencies_005.mean(),
                #     "std" : sufficiencies_005.std()
                # },
                # "comprehensiveness @ 0.05" : {
                #     "mean" : comprehensivenesses_005.mean(),
                #     "std" : comprehensivenesses_005.std()
                # },


                # "sufficiencies @ 0.1" : {
                #     "mean" : sufficiencies_01.mean(),
                #     "std" : sufficiencies_01.std()
                # },
                # "comprehensiveness @ 0.1" : {
                #     "mean" : comprehensivenesses_01.mean(),
                #     "std" : comprehensivenesses_01.std()
                # },

                
                # "sufficiencies @ 0.2" : {
                #     "mean" : sufficiencies_02.mean(),
                #     "std" : sufficiencies_02.std()
                # },
                # "comprehensiveness @ 0.2" : {
                #     "mean" : comprehensivenesses_02.mean(),
                #     "std" : comprehensivenesses_02.std()
                # },
                

                # "sufficiencies @ 0.5" : {
                #     "mean" : sufficiencies_05.mean(),
                #     "std" : sufficiencies_05.std()
                # },
                # "comprehensiveness @ 0.5" : {
                #     "mean" : comprehensivenesses_05.mean(),
                #     "std" : comprehensivenesses_05.std()
                # },


                "sufficiencies @ 1.0" : {
                    "mean" : sufficiencies_10.mean(),
                    "std" : sufficiencies_10.std()
                },
                "comprehensiveness @ 1.0" : {
                    "mean" : comprehensivenesses_10.mean(),
                    "std" : comprehensivenesses_10.std()
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


            description_fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-normal_{normalise}.json"
        else:
            sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
            comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
            
            descriptor[feat_attr] = {
                "sufficiency" : {
                    "mean" : sufficiencies.mean(),
                    "std" : sufficiencies.std()
                },
                "comprehensiveness" : {
                    "mean" : comprehensivenesses.mean(),
                    "std" : comprehensivenesses.std()
                },
            }
            description_fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-normal_{normalise}.json"

    #np.save(detailed_fname, faithfulness_results)
    with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 

    return


def conduct_experiments_noise_(model, data, model_random_seed, std, use_topk, normalise=0): #faithful_method
    ## now to create folder where results will be saved
    fname = os.path.join(os.getcwd(),args["data_dir"],"importance_scores", "")
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()

    ## retrieve original prediction probability
    fname2 = os.path.join(os.getcwd(),args["model_dir"])
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()
    
    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    desired_rationale_length = args.rationale_length


    faithfulness_results = {}
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
                "importance_scores":torch.zeros(batch["input_ids"].squeeze(1).size()),  # baseline, so all important 
                #"add_noise": False,
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
        #print("==>> (original_prediction): ", (original_prediction))
        


        full_text_probs = original_prediction.max(-1) 
        full_text_class = original_prediction.argmax(-1)


        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))


        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:
            only_query_mask=create_only_query_mask_(batch_input_ids=batch["input_ids"], special_tokens=batch["special_tokens"])
            batch["input_ids"] = only_query_mask * original_sentences
        else:
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            batch["input_ids"] = only_query_mask 


        batch["faithful_method"] = "soft_suff" 

        batch["importance_scores"]=torch.zeros(batch["input_ids"].squeeze(1).size()) # 都不重要
        batch["add_noise"]=True  ### 测试!!!!!!!!!!!! zeroout(probably does not matter) and attention is false
        #batch["rationale_mask"]=torch.zeros(batch["input_ids"].size())
        yhat, _  = model(**batch) # 此时 input id 全为o, 做的baseline ---> suff(x, y', 0)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        ) # Suff(x, ˆ y, 0) , no rationales to compare
        # print("==>> (suff_y_zero): ")

        # print("==>> (suff_y_zero): ")
        # print("==>> (suff_y_zero): ", (suff_y_zero))
        # print("==>> 1-(suff_y_zero): ", (1-suff_y_zero))


        for _j_, annot_id in enumerate(batch["annotation_id"]):
                    faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                    faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                    for feat in feat_name_dict:
                        faithfulness_results[annot_id][feat] = {}
        
        

        for feat_name in feat_name_dict:#feat_name_dict: #"ig" ,"lime", "deeplift", "deepliftshap", 

            feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            
            for _i_, rationale_length in enumerate([1.0]): 
                #print(' ---------------- >', rationale_length)

                # if args.query:

                #     rationale_mask = create_rationale_mask_(
                #     importance_scores = feat_score, 
                #     no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                #     method = 'topk',
                #     batch_input_ids = original_sentences,
                #     special_tokens = batch["special_tokens"],
                # )
                # else:
                #     rationale_mask = create_rationale_mask_(
                #     importance_scores = feat_score, 
                #     no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                #     method = 'topk',
                #     special_tokens = batch["special_tokens"],
                # )

                soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                model = model, 
                original_sentences = original_sentences, 
                #rationale_mask = rationale_mask, 
                inputs = batch, 
                full_text_probs = full_text_probs,   
                full_text_class = full_text_class,  
                rows = rows,
                suff_y_zero = suff_y_zero, 
                importance_scores = feat_score,
                use_topk=use_topk,
                normalise=normalise,
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
                use_topk=use_topk,
                only_query_mask = only_query_mask,
                normalise=normalise,
                )

            #'''
                # print(' ------------SOFT  SUFF  ----------')
                # print(soft_suff, soft_suff_probs)
                # print(' ------------SOFT  COMP  ----------')
                # print(soft_comp, soft_comp_probs)

                suff_aopc[:,_i_] = soft_suff
                comp_aopc[:,_i_] = soft_comp

  
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    faithfulness_results[annot_id][feat_name][f"sufficiency @ {rationale_length}"] = soft_suff[_j_]
                    faithfulness_results[annot_id][feat_name][f"comprehensiveness @ {rationale_length}"] = soft_comp[_j_]
                    faithfulness_results[annot_id][feat_name][f"masked R probs (comp) @ {rationale_length}"] = soft_comp_probs[_j_].astype(np.float64)
                    faithfulness_results[annot_id][feat_name][f"only R probs (suff) @ {rationale_length}"] = soft_suff_probs[_j_].astype(np.float64)
                    
                    
                    if _i_ == len(rationale_ratios)-1:
                        faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : suff_aopc[_j_]
                                                                        }
                        faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : comp_aopc[_j_]
                                                                        }

            #quit()
        
        pbar.update(data.batch_size)   

    detailed_fname = args["evaluation_dir"] + f"NOISE-std{std}_faithfulness-scores-normal_{normalise}.npy"
    
    np.save(detailed_fname, faithfulness_results)
            
            
            
    descriptor = {}

    # filling getting averages
    for feat_attr in feat_name_dict: #"gradientshap", "lime","deepliftshap",   [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
        
        if use_topk: # 0.05, 0.1, 0.2, 0.5]
            # sufficiencies_001 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.01"] for k in faithfulness_results.keys()])
            # comprehensivenesses_001 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.01"] for k in faithfulness_results.keys()])

            # sufficiencies_002 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.02"] for k in faithfulness_results.keys()])
            # comprehensivenesses_002 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.02"] for k in faithfulness_results.keys()])

            # sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.05"] for k in faithfulness_results.keys()])
            # comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

            # sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.1"] for k in faithfulness_results.keys()])
            # comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

            # sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.2"] for k in faithfulness_results.keys()])
            # comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

            # sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
            # comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

            # sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
            # comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

            sufficiencies_10 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 1.0"] for k in faithfulness_results.keys()])
            comprehensivenesses_10 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 1.0"] for k in faithfulness_results.keys()])

            #aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
            #aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])
            
            descriptor[feat_attr] = {
                # "sufficiencies @ 0.01" : {
                #     "mean" : sufficiencies_001.mean(),
                #     "std" : sufficiencies_001.std()
                # },
                # "comprehensiveness @ 0.01" : {
                #     "mean" : comprehensivenesses_001.mean(),
                #     "std" : comprehensivenesses_001.std()
                # },


                # "sufficiencies @ 0.02" : {
                #     "mean" : sufficiencies_002.mean(),
                #     "std" : sufficiencies_002.std()
                # },
                # "comprehensiveness @ 0.02" : {
                #     "mean" : comprehensivenesses_002.mean(),
                #     "std" : comprehensivenesses_002.std()
                # },

                # "sufficiencies @ 0.05" : {
                #     "mean" : sufficiencies_005.mean(),
                #     "std" : sufficiencies_005.std()
                # },
                # "comprehensiveness @ 0.05" : {
                #     "mean" : comprehensivenesses_005.mean(),
                #     "std" : comprehensivenesses_005.std()
                # },


                # "sufficiencies @ 0.1" : {
                #     "mean" : sufficiencies_01.mean(),
                #     "std" : sufficiencies_01.std()
                # },
                # "comprehensiveness @ 0.1" : {
                #     "mean" : comprehensivenesses_01.mean(),
                #     "std" : comprehensivenesses_01.std()
                # },

                
                # "sufficiencies @ 0.2" : {
                #     "mean" : sufficiencies_02.mean(),
                #     "std" : sufficiencies_02.std()
                # },
                # "comprehensiveness @ 0.2" : {
                #     "mean" : comprehensivenesses_02.mean(),
                #     "std" : comprehensivenesses_02.std()
                # },
                

                # "sufficiencies @ 0.5" : {
                #     "mean" : sufficiencies_05.mean(),
                #     "std" : sufficiencies_05.std()
                # },
                # "comprehensiveness @ 0.5" : {
                #     "mean" : comprehensivenesses_05.mean(),
                #     "std" : comprehensivenesses_05.std()
                # },


                "sufficiencies @ 1.0" : {
                    "mean" : sufficiencies_10.mean(),
                    "std" : sufficiencies_10.std()
                },
                "comprehensiveness @ 1.0" : {
                    "mean" : comprehensivenesses_10.mean(),
                    "std" : comprehensivenesses_10.std()
                },


            #     "AOPC - sufficiency" : {
            #         "mean" : aopc_suff.mean(),
            #         "std" : aopc_suff.std()
            #     },
            #     "AOPC - comprehensiveness" : {
            #         "mean" : aopc_comp.mean(),
            #         "std" : aopc_comp.std()
            #     }
            }        


            description_fname = args["evaluation_dir"] + f"NOISE-std{std}_faithfulness-scores-normal_{normalise}.json"
        else:
            sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
            comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
            
            descriptor[feat_attr] = {
                "sufficiency" : {
                    "mean" : sufficiencies.mean(),
                    "std" : sufficiencies.std()
                },
                "comprehensiveness" : {
                    "mean" : comprehensivenesses.mean(),
                    "std" : comprehensivenesses.std()
                },
            }
            description_fname = args["evaluation_dir"] + f"NOISE-std{std}_faithfulness-scores-normal_{normalise}.json"


    #np.save(detailed_fname, faithfulness_results)
    with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 


    return


def conduct_experiments_attention_(model, data, model_random_seed, use_topk, normalise=0): #faithful_method

    fname = os.path.join(os.getcwd(),args["data_dir"],"importance_scores","")
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores_{model_random_seed}.npy"
    importance_scores = np.load(fname, allow_pickle = True).item()

    ## retrieve original prediction probability
    fname2 = os.path.join(os.getcwd(),args["model_dir"])
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()
   
    desc = 'faithfulness evaluation -> id'
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    desired_rationale_length = args.rationale_length


    faithfulness_results = {}
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

        full_text_probs = original_prediction.max(-1) 
        full_text_class = original_prediction.argmax(-1)

        rows = np.arange(batch["input_ids"].size(0))
        
        if args.query:
            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"],
            )
            batch["input_ids"] = only_query_mask * original_sentences
        else:
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            batch["input_ids"] = only_query_mask


        batch["faithful_method"] = "soft_suff"
        batch["importance_scores"]=torch.zeros(batch["input_ids"].squeeze(1).size())
        batch["rationale_mask"]=torch.zeros(batch["input_ids"].size())
        batch["add_noise"]=True  ### 测试点, noise定用true
        yhat, _  = model(**batch) # 此时 input id 全为o, 做的baseline ---> suff(x, y', 0)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )



        batch["add_noise"]=True

        for _j_, annot_id in enumerate(batch["annotation_id"]):
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            for feat in feat_name_dict:
                    faithfulness_results[annot_id][feat] = {}
        
        

        for feat_name in feat_name_dict: # ,"lime", , "deepliftshap", 

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

            for _i_, rationale_length in enumerate(rationale_ratios):


                if rationale_length == 1.0: 
                    rationale_mask= torch.ones(batch["input_ids"].size())

                    soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                        model = model, 
                        original_sentences = original_sentences, 
                        #rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        importance_scores = feat_score,
                        use_topk=False,
                        normalise=normalise,
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
                        use_topk=False,
                        only_query_mask=only_query_mask,
                        normalise=normalise,
                    )
                    
                    
                else:    
                    pass
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
                            special_tokens = batch["special_tokens"],
                        )


                    soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                        model = model, 
                        original_sentences = original_sentences, 
                        #rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        importance_scores = feat_score,
                        use_topk=True,
                        normalise=normalise,
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
                        use_topk=True,
                        only_query_mask=only_query_mask,
                        normalise=normalise,
                    )
                

                suff_aopc[:,_i_] = soft_suff
                comp_aopc[:,_i_] = soft_comp

                
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    # faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                    # faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                
                    faithfulness_results[annot_id][feat_name][f"sufficiency @ {rationale_length}"] = soft_suff[_j_]
                    faithfulness_results[annot_id][feat_name][f"comprehensiveness @ {rationale_length}"] = soft_comp[_j_]
                    faithfulness_results[annot_id][feat_name][f"masked R probs (comp) @ {rationale_length}"] = soft_comp_probs[_j_].astype(np.float64)
                    faithfulness_results[annot_id][feat_name][f"only R probs (suff) @ {rationale_length}"] = soft_suff_probs[_j_].astype(np.float64)
                

                    
                    if _i_ == len(rationale_ratios)-1:
                        faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                        "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : suff_aopc[_j_]
                                                                        }
                        faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                        "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                        "per ratio" : comp_aopc[_j_]
                                                                        }

              
    pbar.update(data.batch_size)
        
    detailed_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-normal_{normalise}.npy"
    np.save(detailed_fname, faithfulness_results)

            
    descriptor = {}

        # filling getting averages
    for feat_attr in feat_name_dict: #"gradientshap", "lime","deepliftshap",   [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
        
        if use_topk: # 0.05, 0.1, 0.2, 0.5]
            sufficiencies_001 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.01"] for k in faithfulness_results.keys()])
            comprehensivenesses_001 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.01"] for k in faithfulness_results.keys()])

            sufficiencies_002 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.02"] for k in faithfulness_results.keys()])
            comprehensivenesses_002 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.02"] for k in faithfulness_results.keys()])

            sufficiencies_005 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.05"] for k in faithfulness_results.keys()])
            comprehensivenesses_005 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.05"] for k in faithfulness_results.keys()])

            sufficiencies_01 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.1"] for k in faithfulness_results.keys()])
            comprehensivenesses_01 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.1"] for k in faithfulness_results.keys()])

            sufficiencies_02 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.2"] for k in faithfulness_results.keys()])
            comprehensivenesses_02 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.2"] for k in faithfulness_results.keys()])

            sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
            comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

            sufficiencies_05 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 0.5"] for k in faithfulness_results.keys()])
            comprehensivenesses_05 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 0.5"] for k in faithfulness_results.keys()])

            sufficiencies_10 = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ 1.0"] for k in faithfulness_results.keys()])
            comprehensivenesses_10 = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ 1.0"] for k in faithfulness_results.keys()])

            aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
            aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])
            
            descriptor[feat_attr] = {
                "sufficiencies @ 0.01" : {
                    "mean" : sufficiencies_001.mean(),
                    "std" : sufficiencies_001.std()
                },
                "comprehensiveness @ 0.01" : {
                    "mean" : comprehensivenesses_001.mean(),
                    "std" : comprehensivenesses_001.std()
                },


                "sufficiencies @ 0.02" : {
                    "mean" : sufficiencies_002.mean(),
                    "std" : sufficiencies_002.std()
                },
                "comprehensiveness @ 0.02" : {
                    "mean" : comprehensivenesses_002.mean(),
                    "std" : comprehensivenesses_002.std()
                },

                "sufficiencies @ 0.05" : {
                    "mean" : sufficiencies_005.mean(),
                    "std" : sufficiencies_005.std()
                },
                "comprehensiveness @ 0.05" : {
                    "mean" : comprehensivenesses_005.mean(),
                    "std" : comprehensivenesses_005.std()
                },


                "sufficiencies @ 0.1" : {
                    "mean" : sufficiencies_01.mean(),
                    "std" : sufficiencies_01.std()
                },
                "comprehensiveness @ 0.1" : {
                    "mean" : comprehensivenesses_01.mean(),
                    "std" : comprehensivenesses_01.std()
                },

                
                "sufficiencies @ 0.2" : {
                    "mean" : sufficiencies_02.mean(),
                    "std" : sufficiencies_02.std()
                },
                "comprehensiveness @ 0.2" : {
                    "mean" : comprehensivenesses_02.mean(),
                    "std" : comprehensivenesses_02.std()
                },
                

                "sufficiencies @ 0.5" : {
                    "mean" : sufficiencies_05.mean(),
                    "std" : sufficiencies_05.std()
                },
                "comprehensiveness @ 0.5" : {
                    "mean" : comprehensivenesses_05.mean(),
                    "std" : comprehensivenesses_05.std()
                },


                "sufficiencies @ 1.0" : {
                    "mean" : sufficiencies_10.mean(),
                    "std" : sufficiencies_10.std()
                },
                "comprehensiveness @ 1.0" : {
                    "mean" : comprehensivenesses_10.mean(),
                    "std" : comprehensivenesses_10.std()
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


            description_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-normal_{normalise}.json"
        else:
            sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency"] for k in faithfulness_results.keys()])
            comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness"] for k in faithfulness_results.keys()])
            
            descriptor[feat_attr] = {
                "sufficiency" : {
                    "mean" : sufficiencies.mean(),
                    "std" : sufficiencies.std()
                },
                "comprehensiveness" : {
                    "mean" : comprehensivenesses.mean(),
                    "std" : comprehensivenesses.std()
                },
            }
            description_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-normal_{normalise}.json"

    
    #np.save(detailed_fname, faithfulness_results)
    with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 

    return



