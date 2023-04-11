import torch
from torch import nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
import json


softmax = torch.nn.Softmax(dim = 1)

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

    return sufficiency




def normalized_sufficiency_(model, 
                            original_sentences : torch.tensor, rationale_mask : torch.tensor, 
                            inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                            suff_y_zero : np.array, only_query_mask : torch.tensor) -> np.array:

    ## for sufficiency we always keep the rationale (from the document) + query
    ## since '1' represent rationale tokens ---> keep in suff predictions & out in comp predictions
    ## preserve cls 
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1

    mask = rationale_mask + only_query_mask
    mask[mask>1] = mask[mask>1]-1
    assert torch.max(mask).item() <=1
    assert mask.size() == original_sentences.size()
    inputs["input_ids"]  =  mask * original_sentences

    
    yhat, _  = model(**inputs)
    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
    reduced_probs = yhat[rows, full_text_class]

    ## reduced input sufficiency
    suff_y_a = sufficiency_(full_text_probs, reduced_probs)
    

    # return suff_y_a
    suff_y_zero -= 1e-4 ## to avoid nan

    norm_suff = np.maximum(0, (suff_y_a - suff_y_zero) / (1 - suff_y_zero))


    norm_suff = np.clip( norm_suff, a_min = 0, a_max = 1)

    return norm_suff, yhat


def normal_importance(importance_scores, normalise):
    #importance_scores[importance_scores==float('-inf')] = -999  # remove -inf, if version higher than 1.11.0, go for torch.clip
    #print("==>> importance_scores[:,0].shape: ", importance_scores)
    if normalise == 1:
        importance_scores = torch.sigmoid(importance_scores) # 偏大, hover 0.5 SUFF works
    elif normalise == 2:
        importance_scores[torch.isinf(importance_scores)] = -10  #importance_scores -= 1e-4 # modify by cass 1711
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
    elif normalise == 3:
        softmax = torch.nn.Softmax(dim = -1)
        importance_scores =softmax(importance_scores).to(device)
    elif normalise == 4:
        importance_scores[torch.isinf(importance_scores)] = -3
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
    elif normalise == 5:
        importance_scores[torch.isinf(importance_scores)] = -1
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
    elif normalise == 6:
        importance_scores = torch.sigmoid(importance_scores)
        importance_scores[torch.isinf(importance_scores)] = -1
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
    elif normalise == 7:
        importance_scores[torch.isinf(importance_scores)] = -1
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
        importance_scores = torch.sigmoid(importance_scores)
    else:pass
    #print("==>> importance_scores[:,0].shape: ", importance_scores)
    #importance_scores[:,0]=1
    #print("==>> importance_scores[:,0].shape: ", importance_scores)
    #

    return importance_scores


# importance_scores = torch.rand([4,6])



def normalized_sufficiency_soft_(model2, use_topk,
                            original_sentences : torch.tensor, 
                            #rationale_mask : torch.tensor, # by Cass, 这里用上rationale nikos想要的
                            inputs : dict, 
                            full_text_probs : np.array, 
                            full_text_class : np.array, 
                            rows : np.array, 
                            suff_y_zero : np.array,
                            importance_scores: torch.tensor,
                            only_query_mask: torch.tensor,
                            normalise: int,
                            ) -> np.array:

    ######### for sufficiency we always keep the rationale
    ######### since ones represent rationale tokens
#
    ####### 这里可以变动 only for suff
    ######## preserve cls
    # rationale_mask[:,0] = 1
    # ######## preserve sep
    # rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1

    # mask = rationale_mask.to(device) + only_query_mask.to(device)
    # mask[mask>1] = mask[mask>1]-1
    # assert torch.max(mask).item() <=1
    # assert mask.size() == original_sentences.size()
    # inputs["input_ids"]  = (mask * original_sentences).type(torch.int64)

    # else: 
    #     inputs["input_ids"]  =  original_sentences

    inputs["input_ids"] = original_sentences
    inputs["faithful_method"]="soft_suff"          
    #inputs["importance_scores"]=importance_scores 
    inputs["add_noise"] = True                     


    inputs["importance_scores"]= normal_importance(importance_scores, normalise)


    yhat, _  = model2(**inputs)
    

    yhat = torch.softmax(yhat.detach().cpu(), dim = -1).numpy()
    reduced_probs = yhat[rows, full_text_class]

    ## reduced input sufficiency
    suff_y_a = sufficiency_(full_text_probs, reduced_probs)

    # return suff_y_a
    suff_y_zero -= 1e-4 ## to avoid nan
    # 在这之前, soft comp和soft suff 都是一样的 (模型里面不一样), 
    norm_suff = np.maximum(0, (suff_y_a - suff_y_zero) / (1 - suff_y_zero))
    norm_suff = np.clip( norm_suff, a_min = 0, a_max = 1)

    return norm_suff, yhat



def normalized_comprehensiveness_soft_(model2, use_topk,
                                        original_sentences : torch.tensor, 
                                        rationale_mask : torch.tensor, 
                                        inputs : dict, 
                                        full_text_probs : np.array, 
                                        full_text_class : np.array, 
                                        rows : np.array, 
                                        suff_y_zero : np.array,
                                        importance_scores: torch.tensor,
                                        #only_query_mask: torch.tensor,
                                        normalise: int,
                                        ) -> np.array:
    inputs["faithful_method"]="soft_comp"
    inputs["add_noise"] = True
    inputs["input_ids"] = original_sentences

    inputs["importance_scores"]= normal_importance(importance_scores, normalise)
    ##### 进 model 前, rationale 已经因为comp 被删掉了
    yhat, _  = model2(**inputs)
    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()




    reduced_probs = yhat[rows, full_text_class]
    comp_y_a = comprehensiveness_(full_text_probs, reduced_probs)


    suff_y_zero[(1-suff_y_zero)==0] += 0.00001

    norm_comp = np.maximum(0, comp_y_a / (1-suff_y_zero))

    norm_comp = np.clip(norm_comp, a_min = 0, a_max = 1)


    return norm_comp, yhat




def normalized_comprehensiveness_(model, 
                                original_sentences : torch.tensor, 
                                rationale_mask : torch.tensor, 
                                inputs : dict, 
                                full_text_probs : np.array, 
                                full_text_class : np.array, 
                                rows : np.array, 
                                suff_y_zero : np.array,
                                
                                ) -> np.array: #suff_y_zero : np.array, 
    
    ## for comprehensivness we always remove the rationale and keep the rest of the input
    ## since ones represent rationale tokens, invert them and multiply the original input

    rationale_mask = (rationale_mask == 0)

    
    ## preserve cls
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1
    inputs["input_ids"] =  original_sentences * rationale_mask.long().to(device)


    
    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
    reduced_probs = yhat[rows, full_text_class]

     ## reduced input sufficiency
    comp_y_a = comprehensiveness_(full_text_probs, reduced_probs)
    
    suff_y_zero[(1-suff_y_zero)==0] += 0.00001 # avoid denominator = 0 把 等于0 的 加 1额-8, 其他不变
    norm_comp = np.maximum(0, comp_y_a / (1-suff_y_zero))


    norm_comp = np.clip(norm_comp, a_min = 0, a_max = 1)

    return norm_comp, yhat



def comprehensiveness_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)

    return comprehensiveness

