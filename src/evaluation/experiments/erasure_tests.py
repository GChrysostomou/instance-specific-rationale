from pickle import NONE
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


feat_name_dict = {"attention", "scaled attention", "gradients", "ig", "deeplift", "random"}
rationale_ratios = [1.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 

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
        

        for _j_, annot_id in enumerate(batch["annotation_id"]):
                
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            for feat in feat_name_dict:
                faithfulness_results[annot_id][feat] = {}
                
        for feat_name in feat_name_dict: #"ig" ,"lime", "deeplift", "gradientshap",

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

            print('')
            print('')

            print('============================>>>>>>>>>>>>>>>>', feat_name )
            for _i_, rationale_length in enumerate(rationale_ratios):
                print('============================>>>>>>>>>>>>>>>>', rationale_length )

                if args.query:
                    len_tensor = batch["lengths"].clone()
                    length_f = len_tensor.float()
                    temp = length_f * rationale_length
                    tempB = torch.ceil(temp)
                    tempC = tempB.detach().cpu().numpy()

                    rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            # no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            no_of_masked_tokens = tempC,
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

                print("".center(50, "-"))
                print("".center(50, "-"))
                print("".center(50, "-"))
                print("==>> type(suff): ", type(suff), suff)
                print("==>> type(suff_probs): ", type(suff_probs), suff_probs)

                quit()

                comp, comp_probs  = normalized_comprehensiveness_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = rationale_mask, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    #suff_y_zero = suff_y_zero,
                    comp_y_one= 1-suff_y_zero,
                )

                
                
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




def conduct_experiments_zeroout_(model, data, model_random_seed, use_topk):

    
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
                #"importance_scores":torch.ones(batch["input_ids"].squeeze(1).size()),
                #"add_noise": False,
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
        batch["add_noise"]=False
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
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        comp_y_one= 1-suff_y_zero,
                        importance_scores = feat_score,
                        use_topk=False,
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
                        use_topk=False,
                        only_query_mask=only_query_mask,
                    )
                
                
                else:
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
                        comp_y_one= 1-suff_y_zero,
                        importance_scores = feat_score,
                        use_topk=True,
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
                        use_topk=True,
                        only_query_mask=only_query_mask,
                    )

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

    detailed_fname = args["evaluation_dir"] + f"ZEROOUTlimit-faithfulness-scores-detailed.npy"
        #description_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-description.json"
    np.save(detailed_fname, faithfulness_results)
            


    descriptor = {}

    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",   [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
        
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


            description_fname = args["evaluation_dir"] + f"ZEROOUTlimit-faithfulness-scores-description.json"
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
            description_fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-description.json"

    #np.save(detailed_fname, faithfulness_results)
    with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 

    return




def conduct_experiments_noise_(model, data, model_random_seed, std, use_topk): #faithful_method
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
                "std": std,
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

        

        batch["faithful_method"] = "soft_suff"
        batch["importance_scores"]=torch.zeros(batch["input_ids"].squeeze(1).size())
        batch["add_noise"]=False
        yhat, _  = model(**batch) # 此时 input id 全为o, 做的baseline ---> suff(x, y', 0)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        ) # Suff(x, ˆ y, 0) , no rationales to compare


        for _j_, annot_id in enumerate(batch["annotation_id"]):
                    faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                    faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
                    for feat in feat_name_dict:
                        faithfulness_results[annot_id][feat] = {}
        
        
        if use_topk:

            for feat_name in {"random", "attention", "gradients", "scaled attention", "ig","deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 
                feat_score =  batch_from_dict_(
                        batch_data = batch, 
                        metadata = importance_scores, 
                        target_key = feat_name,
                    )

                suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                
                for _i_, rationale_length in enumerate(rationale_ratios):  
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
                        comp_y_one = 1-suff_y_zero,    
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
                        only_query_mask = only_query_mask,
                    )
 
                    
                    suff_aopc[:,_i_] = soft_suff
                    comp_aopc[:,_i_] = soft_comp

                    # if rationale_length == desired_rationale_length:

                    #     sufficiency = soft_suff
                    #     comprehensiveness = soft_comp
                    #     comp_probs_save = soft_comp_probs
                    #     suff_probs_save = soft_suff_probs

                    
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
                

           
        else: # not use topk
            rationale_length = 1
            for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 
                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                # suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
                # comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

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
                    comp_y_one= 1-suff_y_zero,
                    use_topk=use_topk,

                )
                soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    only_query_mask=only_query_mask,
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


                # suff_aopc[:,:] = soft_suff
                # comp_aopc[:,:] = soft_comp

                #if rationale_length == desired_rationale_length:

                # suff_aopc[:,_i_] = soft_suff
                # comp_aopc[:,_i_] = soft_comp


                    
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                   faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency" : soft_suff[_j_],
                        f"comprehensiveness" : soft_comp[_j_],
                        f"masked R probs (comp)" : soft_comp_probs[_j_].astype(np.float64),
                        f"only R probs (suff)" : soft_suff_probs[_j_].astype(np.float64),
      
                   }
            
            
     
        pbar.update(data.batch_size)


    if use_topk:
        detailed_fname = args["evaluation_dir"] + f"NOISElimit-faithfulness-scores-detailed.npy"
        #description_fname = args["evaluation_dir"] + f"ATTENTIONlimit-faithfulness-scores-description.json"
    else:
        detailed_fname = args["evaluation_dir"] + f"NOISE-faithfulness-scores-detailed.npy"
        #description_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-description.json"

    np.save(detailed_fname, faithfulness_results)
            
            
            
    descriptor = {}

    # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",   [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
        
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


            description_fname = args["evaluation_dir"] + f"NOISElimit-faithfulness-scores-description.json"
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
            description_fname = args["evaluation_dir"] + f"NOISE-faithfulness-scores-description.json"


    #np.save(detailed_fname, faithfulness_results)
    with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 


    return



def conduct_experiments_attention_(model, data, model_random_seed, use_topk): #faithful_method

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
                "importance_scores":torch.zeros(batch["input_ids"].squeeze(1).size()),  # baseline, so all not important 
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

        rows = np.arange(batch["input_ids"].size(0))
        
        ## now measuring baseline comprehensiven for all 1 rationale mask
        ## no rationale should be more comprehensive than an all-one rationale
        ## "importance_scores":torch.ones(batch["input_ids"].squeeze(1).size()),  # take all --> no info
        # batch["faithful_method"] = "soft_comp"
        # batch["importance_scores"]=torch.ones(batch["input_ids"].squeeze(1).size())
        # batch["add_noise"]=False
        # yhat, _  = model(**batch)
        # yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        # reduced_probs = yhat[rows, full_text_class]
        # comp_y_one = comprehensiveness_(
        #     full_text_probs, 
        #     reduced_probs
        # )
        ## now measuring baseline sufficiency for all 0 rationale mask
        ## no rationale should be much less sufficient than all-zero rationale  # keep none --> no info
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
        batch["add_noise"]=False
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
        
        
        if use_topk:

            for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", "deeplift"}: # ,"lime", , "deepliftshap", 
                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

                for _i_, rationale_length in enumerate(rationale_ratios):
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
                    soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                                                    model = model, 
                                                    original_sentences = original_sentences, 
                                                    rationale_mask = rationale_mask,
                                                    inputs = batch, 
                                                    full_text_probs = full_text_probs,   
                                                    full_text_class = full_text_class,  
                                                    rows = rows,    
                                                    importance_scores = feat_score,
                                                    comp_y_one= 1-suff_y_zero,
                                                    #suff_y_zero = suff_y_zero,
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
                                                    only_query_mask=only_query_mask,
                                                    use_topk=use_topk,
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
                    

                        # print(_i_)
                        
                        if _i_ == len(rationale_ratios)-1:
                            faithfulness_results[annot_id][feat_name]["sufficiency aopc"] = {
                                                                            "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                            "per ratio" : suff_aopc[_j_]
                                                                            }
                            faithfulness_results[annot_id][feat_name]["comprehensiveness aopc"] = {
                                                                            "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios)),
                                                                            "per ratio" : comp_aopc[_j_]
                                                                            }

        
        else:  ## use the whole, not only topk
            rationale_length = 1

            for feat_name in {"random", "attention", "scaled attention", "gradients", "ig", "deeplift"}: # ,"lime", , "deepliftshap", 

                feat_score =  batch_from_dict_(batch_data = batch, 
                                                metadata = importance_scores, 
                                                target_key = feat_name,
                                            )


                # suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
                # comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

                # if args.query:

                #     rationale_mask = create_rationale_mask_(
                #         importance_scores = feat_score, 
                #         no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                #         #method = rationale_type,
                #         batch_input_ids = original_sentences,
                #         special_tokens = batch["special_tokens"],
                #     )
                # else:
                #     rationale_mask = create_rationale_mask_(
                #         importance_scores = feat_score, 
                #         no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                #         #method = rationale_type
                #     )

                # 在这里面决定用 comprehensive 还是 sufficiency
                soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
                    rationale_mask = None,
                    inputs = batch, 
                    full_text_probs = full_text_probs,   
                    full_text_class = full_text_class,  
                    rows = rows,    
                    importance_scores = feat_score,
                    comp_y_one= 1-suff_y_zero,
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

                # suff_aopc[:,_i_] = soft_suff
                # comp_aopc[:,_i_] = soft_comp

                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency" : soft_suff[_j_],
                        f"comprehensiveness" : soft_comp[_j_],
                        f"masked R probs (comp)" : soft_comp_probs[_j_].astype(np.float64),
                        f"only R probs (suff)" : soft_suff_probs[_j_].astype(np.float64),
      
                    }
            
        pbar.update(data.batch_size)

    if use_topk:
        detailed_fname = args["evaluation_dir"] + f"ATTENTIONlimit-faithfulness-scores-detailed.npy"
    else:
        detailed_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-detailed.npy"

    np.save(detailed_fname, faithfulness_results)

            
    descriptor = {}

        # filling getting averages
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "deeplift"}: #"gradientshap", "lime","deepliftshap",   [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
        
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


            description_fname = args["evaluation_dir"] + f"ATTENTIONlimit-faithfulness-scores-description.json"
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
            description_fname = args["evaluation_dir"] + f"ATTENTION-faithfulness-scores-description.json"

    
    #np.save(detailed_fname, faithfulness_results)
    with open(description_fname, 'w') as file:
            json.dump(descriptor,file,indent = 4) 

    return



