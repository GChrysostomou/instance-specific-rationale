# """
# contains functions for helping with the loading, processing, description and preparation of the datasets
# """

import pandas as pd
import numpy as np
import math
import json 
import torch

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))
    
def wpiece2word(tokenizer, sentence, weights, print_err = False):  

    """
    converts word-piece ids to words and
    importance scores/weights for word-pieces to importance scores/weights
    for words by aggregating them
    """

    tokens = tokenizer.convert_ids_to_tokens(sentence)

    new_words = {}
    new_score = {}

    position = 0

    for i in range(len(tokens)):

        word = tokens[i]
        score = weights[i]

        if "##" not in word:
            
            position += 1
            new_words[position] = word
            new_score[position] = score
            
        else:
            
            new_words[position] += word.split("##")[1]
            new_score[position] += score

    return np.asarray(list(new_words.values())), np.asarray(list(new_score.values()))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mask_topk(sentences, scores, length_to_mask):

    length_to_mask = torch.topk(scores, length_to_mask)
    mask = torch.ones(sentences.shape).to(device)
    mask = mask.scatter_(-1,  length_to_mask[1], 0)

    return sentences * mask.long()

def mask_contigious(sentences, scores, length_to_mask):
    
    length_to_mask = length_to_mask
    
    if len(sentences.size()) < 2: ## single instance

        ngram = torch.stack([scores[k:k + length_to_mask] for k in range(len(scores) - length_to_mask + 1)])
        indxs = [torch.arange(j, j+length_to_mask) for j in range(len(scores) - length_to_mask + 1)]
        
        indexes = indxs[ngram.sum(-1).argmax()]


    else:

        indexes = torch.zeros(scores.size(0), length_to_mask).to(device)

        for i in range(scores.size(0)):

            score = scores[i]
            
            ngram = torch.stack([score[k:k + length_to_mask] for k in range(len(score) - length_to_mask + 1)])
            indxs = [torch.arange(j, j+length_to_mask) for j in range(len(score) - length_to_mask + 1)]
            
            indexes[i] = indxs[ngram.sum(-1).argmax()]

    mask = torch.ones(sentences.shape).to(device)
    mask = mask.scatter_(-1,  indexes.long().to(device), 0)
    
    return sentences * mask.long()

def batch_from_dict(batch_data, metadata, target_key = "original prediction", extra_layer = None):

    new_tensor = []

    for _id_ in batch_data["annotation_id"]:
        
        ## for double nested dics
        if extra_layer :

            new_tensor.append(
                metadata[_id_][extra_layer][target_key]
            )
           
        else:

            new_tensor.append(
                metadata[_id_][target_key]
            )


    
    return torch.tensor(new_tensor).to(device)


def create_rationale_mask_(
        importance_scores : torch.tensor, 
        no_of_masked_tokens : np.ndarray,
        method : str = "topk"
    ):

    rationale_mask = []

    for _i_ in range(importance_scores.size(0)):
        
        score = importance_scores[_i_]
        tokens_to_mask = int(no_of_masked_tokens[_i_])
        
        ## if contigious or not a unigram (unigram == topk of 1)
        if method == "contigious" and tokens_to_mask > 1:

            top_k = contigious_indxs_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )
        
        else:

            top_k = topk_indxs_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )

        ## create the instance specific mask
        ## 1 represents the rationale :)
        ## 0 represents tokens that we dont care about :'(
        mask = torch.zeros(score.shape).to(device)
        mask = mask.scatter_(-1,  top_k.to(device), 1).long()

        rationale_mask.append(mask)

    rationale_mask = torch.stack(rationale_mask).to(device)

    return rationale_mask

## used for preserving queries
def create_only_query_mask_(batch_input_ids : torch.tensor, special_tokens : dict):

    query_mask = []

    for seq in batch_input_ids:
        
        only_query_mask = torch.zeros(seq.shape).to(device)

        sos_eos = torch.where(seq == special_tokens["sep_token_id"][0].item())[0]
        seq_length = sos_eos[0] + 1 
        query_end = sos_eos[1]

        only_query_mask[seq_length: query_end+1] = 1 

        query_mask.append(only_query_mask)

    query_mask = torch.stack(query_mask).to(device)

    return query_mask.long()

def contigious_indxs_(importance_scores, tokens_to_mask):

    ngram = torch.stack([importance_scores[i:i + tokens_to_mask] for i in range(len(importance_scores) - tokens_to_mask + 1)])
    indxs = [torch.arange(i, i+tokens_to_mask) for i in range(len(importance_scores) - tokens_to_mask + 1)]
    top_k = indxs[ngram.sum(-1).argmax()]

    return top_k

def topk_indxs_(importance_scores, tokens_to_mask):

    top_k = torch.topk(importance_scores, tokens_to_mask).indices

    return top_k
