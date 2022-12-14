import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
import json

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class uncertainty_metrics:

    def __init__(self, data : dict, save_dir : str = None, variable : bool = False) -> dict:

        y_prob, y_true = zip(*[(x["predicted"], float(x["actual"])) for x in data.values()])
        # self.y_prob = np.asarray(y_prob)
        self.y_prob = np.asarray([softmax(x) for x in y_prob])
        self.y_true = np.asarray(y_true)
        self.save_dir = save_dir

    def ece(self, n_bins : int =10):

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        confidences, predictions = np.max(self.y_prob, axis=1), np.argmax(self.y_prob, axis=1)
        accuracies = np.equal(predictions, self.y_true)
        ece, refinement = 0.0, 0.0

        bin = 0
        bin_stats = {}
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin += 1
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower) * np.less_equal(confidences, bin_upper)
            prop_in_bin = np.mean(in_bin*1.0)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.absolute(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                refinement += (accuracy_in_bin * (1 - accuracy_in_bin)) * prop_in_bin
                # refinement += (accuracy_in_bin * (1-accuracy_in_bin)) * prop_in_bin
            else:
                avg_confidence_in_bin = 0.
                accuracy_in_bin = 0.
            bin_stats[bin] = {'acc': float(round(accuracy_in_bin,3)), 'conf': float(round(avg_confidence_in_bin,3))}

        fname = self.save_dir + "_ece-stats.json"

        with open(fname, 'w') as file:
            json.dump(
                {'ece': ece, "refinement": refinement, "bins": bin_stats},
                file,
                indent = 4
            ) 

        return {'ece': ece, "refinement": refinement, "bins": bin_stats}

    def entropy_(self,return_vec : bool =False) -> dict:
        ent = -1.0 * np.sum(np.multiply(self.y_prob, np.log(self.y_prob + np.finfo(float).eps)), axis=1) / np.log(2)
        ent_mean = np.mean(ent)
        ent_var = np.var(ent)
        if return_vec:
            return ent

        fname = self.save_dir + "_ece-stats.json"

        with open(fname, 'w') as file:
            json.dump(
                {'mean': float(round(ent_mean,3)), 'var': float(round(ent_var,3))},
                file,
                indent = 4
            ) 

        return {'mean': float(round(ent_mean,3)), 'var': float(round(ent_var,3))}

"""
divergences
"""

def kl_div_loss(p : torch.tensor, q : torch.tensor) -> torch.tensor:

    # adding 1e-10 for 0 to avoid "inf"
    log_p = torch.log(p + 1e-10)
    log_q = torch.log(q + 1e-10)
    kld = p * (log_p - log_q.float())

    return kld.sum(-1)

def jsd(p : torch.tensor, q : torch.tensor) -> torch.tensor:
    mean = 0.5 * (p + q)
    jsd_val = 0.5 * (kl_div_loss(p, mean) + kl_div_loss(q, mean))

    return jsd_val

def perplexity(p : torch.tensor, q : torch.tensor) -> torch.tensor:

    return torch.exp(nn.CrossEntropyLoss()(q, p.argmax(-1)))

def simple_diff(p : torch.tensor, q : torch.tensor) -> torch.tensor:

    rows = torch.arange(p.size(0)).to(device)

    return p.max(-1)[0] - q[rows, p.argmax(-1)]


"""
Faithfulness metrics
"""

def sufficiency_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    sufficiency = 1 - np.maximum(0, full_text_probs - reduced_probs)
    # print('=====full_text_probs====')
    # print(full_text_probs)
    # print('=====reduced_probs====')
    # print(reduced_probs)
    # print('=====sufficiency====')
    # print(sufficiency)
    # quit()

    return sufficiency

def normalized_sufficiency_(model, 
                            original_sentences : torch.tensor, 
                            rationale_mask : torch.tensor, 
                            inputs : dict, 
                            full_text_probs : np.array, 
                            full_text_class : np.array, 
                            rows : np.array, 
                            suff_y_zero : np.array, 
                            only_query_mask : torch.tensor) -> np.array:

    ## for sufficiency we always keep the rationale
    ## since ones represent rationale tokens
    ## preserve cls
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1
    
    inputs["input_ids"]  =  (rationale_mask + only_query_mask) * original_sentences

    yhat, _  = model(**inputs) # wrong

    yhat = torch.softmax(yhat.detach().cpu(), dim = -1).numpy()

    reduced_probs = yhat[rows, full_text_class]

    ## reduced input sufficiency
    suff_y_a = sufficiency_(full_text_probs, reduced_probs)

    # return suff_y_a
    suff_y_zero -= 1e-4 ## to avoid nan

    norm_suff = np.maximum(0, (suff_y_a - suff_y_zero) / (1 - suff_y_zero))

    norm_suff = np.clip( norm_suff, a_min = 0, a_max = 1)

    return norm_suff, yhat

def comprehensiveness_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)

    return comprehensiveness

def normalized_comprehensiveness_(model, original_sentences : torch.tensor, 
                                    rationale_mask : torch.tensor, 
                                  inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                                  suff_y_zero : np.array) -> np.array: #suff_y_zero : np.array, 
    
    ## for comprehensivness we always remove the rationale and keep the rest of the input
    ## since ones represent rationale tokens, invert them and multiply the original input
    rationale_mask = (rationale_mask == 0)
    ## preserve cls
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1

    inputs["input_ids"] =  original_sentences * rationale_mask.long()
    
    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

    reduced_probs = yhat[rows, full_text_class]

     ## reduced input sufficiency
    comp_y_a = comprehensiveness_(full_text_probs, reduced_probs)

    # return comp_y_a
    suff_y_zero -= 1e-6 # to avoid nan

    ## 1 - suff_y_0 == comp_y_1
    norm_comp = np.maximum(0, comp_y_a / (1 - suff_y_zero))

    #norm_comp = np.maximum(0, comp_y_a / comp_y_zero)

    norm_comp = np.clip(norm_comp, a_min = 0, a_max = 1)

    return norm_comp, yhat


def normalized_sufficiency_soft_(model, use_topk,
                            original_sentences : torch.tensor, 
                            rationale_mask : torch.tensor, # by Cass, 这里用上rationale nikos想要的
                            inputs : dict, 
                            full_text_probs : np.array, 
                            full_text_class : np.array, 
                            rows : np.array, 
                            suff_y_zero : np.array,
                            importance_scores: torch.tensor,
                            only_query_mask: torch.tensor,
                            ) -> np.array:

    # for sufficiency we always keep the rationale
    # since ones represent rationale tokens
    # preserve cls

    if use_topk:
        ## preserve cls
        rationale_mask[:,0] = 1
        ## preserve sep
        rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1
        inputs["input_ids"]  =  rationale_mask[:,:original_sentences.size(1)] * original_sentences

    else: 
        inputs["input_ids"]  =  original_sentences

    
    inputs["faithful_method"]="soft_suff"           # for soft, by cass
    inputs["importance_scores"]=importance_scores   # for soft, by cass
    inputs["add_noise"] = True                      # for soft, by cass
    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat.detach().cpu(), dim = -1).numpy()

    reduced_probs = yhat[rows, full_text_class]

    ## reduced input sufficiency
    suff_y_a = sufficiency_(full_text_probs, reduced_probs)

    # return suff_y_a
    suff_y_zero -= 1e-4 ## to avoid nan

    norm_suff = np.maximum(0, (suff_y_a - suff_y_zero) / (1 - suff_y_zero))
    norm_suff = np.clip( norm_suff, a_min = 0, a_max = 1)

    return norm_suff, reduced_probs


def normalized_comprehensiveness_soft_(model, use_topk,
                                    original_sentences : torch.tensor, 
                                  inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                                  suff_y_zero : np.array,
                                  importance_scores: torch.tensor,
                                  rationale_mask : torch.tensor, 
                                  ) -> np.array:
    
    if use_topk:
        # for comprehensivness we always remove the rationale and keep the rest of the input
        # since ones represent rationale tokens, invert them and multiply the original input
        rationale_mask = (rationale_mask == 0)
        ## preserve cls
        rationale_mask[:,0] = 1
        ## preserve sep
        rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1
        inputs["input_ids"] =  original_sentences * rationale_mask.long()
    else:
        inputs["input_ids"] =  original_sentences

        
    inputs["faithful_method"]="soft_comp"
    inputs["importance_scores"]=importance_scores
    inputs["add_noise"] = True
    
    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

    reduced_probs = yhat[rows, full_text_class]

     ## reduced input sufficiency
    comp_y_a = comprehensiveness_(full_text_probs, reduced_probs)

    # return comp_y_a
    suff_y_zero -= 1e-4 # to avoid nan

    ## 1 - suff_y_0 == comp_y_1
    norm_comp = np.maximum(0, comp_y_a / (1 - suff_y_zero))

    norm_comp = np.clip(norm_comp, a_min = 0, a_max = 1)

    return norm_comp, yhat
