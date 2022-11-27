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

from src.common_code.useful_functions import batch_from_dict_, create_only_query_mask_ # batch_from_dict --> batch_from_dict_
from src.common_code.metrics import normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_

from sklearn.metrics import classification_report

## == conduct_experiments
# def conduct_tests_(model, data, split, model_random_seed):

#     """
#         Info: computes the average fraction of tokens required to cause a decision flip (prediction change)
#         Input:
#             model : pretrained model
#             data : torch.DataLoader loaded data
#             save_path : path to save the results
#         Output:
#             saves the results to a csv file under the save_path
#     """

#     desc = f'conducting faithfulness tests on -> {split}'
    
#     pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)

#     ## load rationale masks
#     fname = os.path.join(
#         os.getcwd(),
#         args["extracted_rationale_dir"],
#         args["thresholder"],
#         f"{split}-rationale_metadata.npy"
#     )
#     print('load rationale masks at: ', fname)
    
#     ## retrieve importance scores
#     rationale_metadata = np.load(fname, allow_pickle = True).item()
#     #print('retrieve importance scores: ', rationale_metadata)

#     faithfulness_scores = {}

#     for i, batch in enumerate(data):
#         if i == 0:
#             print('the first batch in data for reference: ')
#             print(batch)
#         model.eval()
#         model.zero_grad()

#         batch = {
#                 "annotation_id" : batch["annotation_id"],
#                 "input_ids" : batch["input_ids"].squeeze(1).to(device),
#                 "lengths" : batch["lengths"].to(device),
#                 "labels" : batch["label"].to(device),
#                 "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
#                 "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
#                 "query_mask" : batch["query_mask"].squeeze(1).to(device),
#                 "special_tokens" : batch["special tokens"],
#                 "retain_gradient" : False ## we do not need it
#             }
            
#         assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
#         yhat, _ =  model(**batch)

#         ## retrieve original full text logits and converts to probs
#         original_prediction = batch_from_dict_(
#             batch_data = batch,
#             metadata = rationale_metadata,
#             target_key =  "original prediction",
#             #extra_layer = None
#         )

#         ## saving our results
#         for annotation_id in batch["annotation_id"]:

#             faithfulness_scores[annotation_id] = {}

#         original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy()

#         full_text_probs = original_prediction.max(-1)
#         full_text_class = original_prediction.argmax(-1)

#         original_sentences = batch["input_ids"].clone()

#         rows = np.arange(original_sentences.size(0))

#         ## now measuring baseline sufficiency for all 0 rationale mask
#         if args.query:

#             only_query_mask=create_only_query_mask_(
#                 batch_input_ids=batch["input_ids"],
#                 special_tokens=batch["special_tokens"]
#             )

#             batch["input_ids"] = only_query_mask * original_sentences

#         else:

#             only_query_mask=torch.zeros_like(batch["input_ids"]).long()

#             batch["input_ids"] = only_query_mask

#         yhat, _  = model(**batch)

#         yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

#         reduced_probs = yhat[rows, full_text_class]

#         ## baseline sufficiency
#         suff_y_zero = sufficiency_(
#             full_text_probs, 
#             reduced_probs
#         )

#         with torch.no_grad():

#             ## start the masking process to measure faithfulness
#             for length_of_rationale in ["fixed"]: #, "variable"
                
#                 ## alias for the variable stuff and for saving
#                 if length_of_rationale == "variable":

#                     var_alias = "var"

#                 else:

#                     var_alias = "fixed"

#                 for feat_attribution_name in ["random", "attention", "scaled attention", "gradients", "ig", "deeplift", "lime"]: #, "--var-feat", "--var-all"
#                         # if feat_attribution_name == "--var-all":
    
#                         #     feat_attribution_name = var_alias + "-len_var-feat_var-type"
                        
#                         # if feat_attribution_name == "--var-feat":

#                         #     feat_attribution_name = var_alias + "-len_var-feat"

#                         # rationale_mask = batch_from_dict_(
#                         #     batch_data = batch,
#                         #     metadata = rationale_metadata,
#                         #     target_key =  f"{length_of_rationale} rationale mask",
#                         #     #extra_layer = feat_attribution_name,
#                         # ) # bug
#                         if args.query:

#                             rationale_mask = create_only_query_mask_(
#                             importance_scores = feat_score, 
#                             no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
#                             method = rationale_type,
#                             batch_input_ids = original_sentences,
#                             special_tokens = batch["special_tokens"],
#                         )

#                         else:

#                             rationale_mask = create_rationale_mask_(
#                             importance_scores = feat_score, 
#                             no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
#                             method = rationale_type
#                         )
#                         ## measuring faithfulness
#                         comp, reduced_probs  = normalized_comprehensiveness_(
#                             model = model, 
#                             original_sentences = original_sentences, 
#                             #rationale_mask = rationale_mask, 
#                             inputs = batch, 
#                             full_text_probs = full_text_probs, 
#                             full_text_class = full_text_class, 
#                             rows = rows,
#                             suff_y_zero = suff_y_zero
#                         )

#                         suff = normalized_sufficiency_(
#                             model = model, 
#                             original_sentences = original_sentences, 
#                             #rationale_mask = rationale_mask, 
#                             inputs = batch, 
#                             full_text_probs = full_text_probs, 
#                             full_text_class = full_text_class, 
#                             rows = rows,
#                             suff_y_zero = suff_y_zero,
#                             only_query_mask=only_query_mask
#                         )
                        
#                         for j_, annot_id in enumerate(batch["annotation_id"]):
                            
#                             if feat_attribution_name == "--var-feat":

#                                 dic_name = feat_attribution_name

#                             else:

#                                 dic_name = var_alias + "-" + feat_attribution_name

#                             faithfulness_scores[annot_id][dic_name] = {
#                                 "comprehensiveness" : float(comp[j_]),
#                                 "sufficiency" : float(suff[j_]),
#                                 "masked prediction probs" : reduced_probs[j_].astype(np.float64).tolist(),
#                                 "full text prediction probs" : original_prediction[j_].astype(np.float64).tolist(),
#                                 "labels" : int(batch["labels"][j_].detach().cpu().item())
#                             }

                    
#         pbar.update(data.batch_size)


#     ## save our results
#     fname = args["evaluation_dir"] + args.thresholder + f"-{split}-faithfulness-metrics.json"

#     with open(fname, 'w') as file:
#             json.dump(
#                 faithfulness_scores,
#                 file,
#                 indent = 4
#             ) 
#     print(' save faithful metrics json file at: ', fname)
#     averages = {}
#     f1s_model = {
#         "f1 macro avg - model labels" : {},
#         "f1 macro avg - actual labels" : {}
#     }

#     random_annot_id = batch["annotation_id"][0]

#     feat_attributions = faithfulness_scores[random_annot_id].keys()

#     for key in feat_attributions:

#             averages[key] = {}

#             for metric in {"comprehensiveness", "sufficiency"}:

#                 averages[key][metric] = {
#                     "mean" : np.asarray([v[key][metric] for k,v in faithfulness_scores.items()]).mean(),
#                     "std" : np.asarray([v[key][metric] for k,v in faithfulness_scores.items()]).std()
#                 } 


#             masked_prediction_probs = [np.asarray(faithfulness_scores[k][key]["masked prediction probs"]).argmax() for k in faithfulness_scores.keys()]
#             full_text__prediction_probs = [np.asarray(faithfulness_scores[k][key]["full text prediction probs"]).argmax() for k in faithfulness_scores.keys()]
#             true_labels = [faithfulness_scores[k][key]["labels"] for k in faithfulness_scores.keys()]

#             out_model = classification_report(full_text__prediction_probs, masked_prediction_probs, output_dict = True)["macro avg"]["f1-score"]
#             out_labels = classification_report(true_labels, masked_prediction_probs, output_dict = True)["macro avg"]["f1-score"]

#             f1s_model["f1 macro avg - model labels"][key] = round(out_model * 100, 3)
#             f1s_model["f1 macro avg - actual labels"][key] = round(out_labels * 100, 3)

#     ## save descriptors
#     fname = args["evaluation_dir"] + args.thresholder + f"-{split}-faithfulness-metrics-description.json"

#     with open(fname, 'w') as file:
#             json.dump(
#                 averages,
#                 file,
#                 indent = 4
#             ) 

#     ## save f1s
#     fname = args["evaluation_dir"] + args.thresholder + f"-{split}-f1-metrics-description.json"


#     with open(fname, 'w') as file:
#             json.dump(
#                 f1s_model,
#                 file,
#                 indent = 4
#             ) 

#     return


def conduct_tests_(model, data, model_random_seed):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores-{model_random_seed}.npy"

    ## retrieve importance scores
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
        rows = np.arange(batch["input_ids"].size(0))

        original_sentences = batch["input_ids"].clone().detach()

        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

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

                        rationale_mask = create_only_query_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = rationale_type,
                            batch_input_ids = original_sentences,
                            special_tokens = batch["special_tokens"],
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
                        suff_y_zero = suff_y_zero
                    )

                    suff, suff_probs = normalized_sufficiency_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero
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
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "gradientshap",
            "deeplift"}: #"ig", "lime",
        
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




def conduct_experiments_zeroout_(model, data, model_random_seed,faithful_method, set):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["extracted_rationale_dir"],
        args["thresholder"],
        #"importance_scores",
        ""
    )
    os.makedirs(fname, exist_ok = True)
    # fname = f"{fname}test_importance_scores-{model_random_seed}.npy"
    fname = f"{fname}test-rationale_metadata.npy"
    ## retrieve importance scores
    importance_scores = np.load(fname, allow_pickle = True).item()

    ## retrieve original prediction probability
    fname2 = os.path.join(
        os.getcwd(),
        args["model_dir"],
    )
    fname2 = glob.glob(fname2 + f"*output*{model_random_seed}.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    print(fname2)

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
        
        # original_prediction, _ =  model(**batch)
        # original_prediction.max(-1)[0].sum().backward(retain_graph = True)
        original_prediction =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = original_prediction_output, 
                    target_key = "predicted",
                )  # return torch.tensor(new_tensor).to(device)

        ## setting up the placeholder for storing the results
        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))

        original_sentences = batch["input_ids"].clone().detach()

        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

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
            # print(' NOT QUERY task')
            # print('original_sentences', original_sentences)
            only_query_mask=torch.zeros_like(batch["input_ids"]).long()
            # print('only_query_mask', only_query_mask)
            batch["input_ids"] = only_query_mask

        
        yhat, _  = model(**batch) # 此时 input id 全为o, 做的baseline ---> suff(x, y', 0)
        # print('after get the baseline from the first model func, the batch is', batch) #已经是0
        # quit()
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
        
        for feat_name in {"random", "attention", "scaled attention", "gradients", "ig", "gradientshap",
        "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 

            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )

            #  feat_score is get by new_tensor.append(metadata[_id_][target_key])

            suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

            soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                model = model, 
                original_sentences = original_sentences, 
                # 下面改成了 其实是 importance scores --> feat_score
                # rationale_mask = feat_score,
                inputs = batch, 
                full_text_probs = full_text_probs, 
                full_text_class = full_text_class, 
                rows = rows,
                suff_y_zero = suff_y_zero,
                importance_scores = feat_score,
            
            )

            soft_suff, soft_suff_probs = normalized_sufficiency_soft_(
                model = model, 
                original_sentences = original_sentences, 
                # rationale_mask = feat_score, 
                inputs = batch, 
                full_text_probs = full_text_probs, 
                full_text_class = full_text_class, 
                rows = rows,
                suff_y_zero = suff_y_zero,
                importance_scores = feat_score,
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
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "gradientshap", "deeplift"}: #"ig", "lime","deepliftshap",
        
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
    if set != None:
        fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-detailed-" + str(set) + ".npy"
    else: fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-detailed.npy"
    np.save(fname, faithfulness_results)

    ## save descriptors
    if set != None: fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-description" + str(set) + ".json"
    else: fname = args["evaluation_dir"] + f"ZEROOUT-faithfulness-scores-description.json"

    with open(fname, 'w') as file:
        print('  ------------->>>> saved as ', fname)
        json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return



def conduct_experiments_noise_(model, data, model_random_seed,faithful_method, std):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        #"importance_scores",
        ""
    )
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores-{model_random_seed}.npy"
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
    print(' /////////////////// original_prediction_output get test_ ////////')
    print(original_prediction_output.get('test_14171'))
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

        batch["add_noise"] = False # all zero sequence and no noise!!! 
        yhat, _  = model(**batch)  # 此时 input id 完全为0
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()


        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        ) # Suff(x, ˆ y, 0) , no rationales to compare


        for _j_, annot_id in enumerate(batch["annotation_id"]):
            faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
            faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
        
        for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", "gradientshap",
        "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 
            feat_score =  batch_from_dict_(
                batch_data = batch, 
                metadata = importance_scores, 
                target_key = feat_name,
            )
            #  feat_score is get by new_tensor.append(metadata[_id_][target_key])

            suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
            comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)

            # 在这里面决定用 comprehensive 还是 sufficiency
            # 在里面决定要不要加noise, 此处加,前面zeroout的baseline不加
            soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                model = model, 
                original_sentences = original_sentences, 
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
                # rationale_mask = feat_score, 
                inputs = batch, 
                full_text_probs = full_text_probs, 
                full_text_class = full_text_class, 
                rows = rows,
                suff_y_zero = suff_y_zero,
                importance_scores = feat_score,
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
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "gradientshap", "deeplift"}: #"ig", "lime","deepliftshap",
        
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
    fname = args["evaluation_dir"] + f"NOISE-faithfulness-scores-detailed-std_" + str(std) + ".npy"

    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = args["evaluation_dir"] + f"NOISE-faithfulness-scores-description-std_" + str(std) + ".json"

    with open(fname, 'w') as file:
        print('  ------------->>>> saved as ', fname)

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
        args["extracted_rationale_dir"],
        #"importance_scores",
        ""
    )
    os.makedirs(fname, exist_ok = True)
    fname = f"{fname}test_importance_scores-{model_random_seed}.npy"
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
    # print(importance_scores.keys())

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
                # "importance_scores": feat_scores,
                #"faithful_method": faithful_method,
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
        rows = np.arange(batch["input_ids"].size(0))
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        ) # Suff(x, ˆ y, 0) , no rationales to compare



        for rationale_type in {args.thresholder}:

            for _j_, annot_id in enumerate(batch["annotation_id"]):

                    
                faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            
            for feat_name in {"random", "attention", "gradients", "scaled attention", "ig", "gradientshap",
            "deeplift"}: #"ig" ,"lime", "deeplift", "deepliftshap", 

                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                #  feat_score is get by new_tensor.append(metadata[_id_][target_key])

                suff_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], 1], dtype=np.float64)
     
                # 在这里面决定用 comprehensive 还是 sufficiency
                soft_comp, soft_comp_probs  = normalized_comprehensiveness_soft_(
                    model = model, 
                    original_sentences = original_sentences, 
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
                    # rationale_mask = feat_score, 
                    inputs = batch, 
                    full_text_probs = full_text_probs, 
                    full_text_class = full_text_class, 
                    rows = rows,
                    suff_y_zero = suff_y_zero,
                    importance_scores = feat_score,
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
    for feat_attr in {"random", "attention", "scaled attention", "ig", "gradients", "gradientshap", "deeplift"}: #"ig", "lime","deepliftshap",
        
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

