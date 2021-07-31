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


def describe_data_stats(path = str):
    """ 
    returns dataset statistics such as : 
                                        - number of documens
                                        - average sequence length
                                        - average query length (if QA)
    """
    # ensure that datapath ends with "/"
    if path[-1] == "/":pass 
    else: path = path + "/"
    
    descriptions = {"train":{}, "dev":{}, "test":{}}

    train = pd.read_csv(path + "train.csv")
    dev = pd.read_csv(path + "dev.csv")
    test = pd.read_csv(path + "test.csv")

    if args["query"]:

        train = train.rename(columns = {"document":"text"})
        dev = dev.rename(columns = {"document":"text"})
        test = test.rename(columns = {"document":"text"})

    # load data and save them in descriptions dictionary
    
    descriptions["train"]["number_of_docs"] =len(train.text.values)
    descriptions["train"]["ave_doc_length"] =  math.ceil(np.asarray([len(x.split()) for x in train.text.values]).mean())

    descriptions["dev"]["number_of_docs"] =len(dev.text.values)
    descriptions["dev"]["ave_doc_length"] =  math.ceil(np.asarray([len(x.split()) for x in dev.text.values]).mean())
    
    descriptions["test"]["number_of_docs"] =len(test.text.values)
    descriptions["test"]["ave_doc_length"] =  math.ceil(np.asarray([len(x.split()) for x in test.text.values]).mean())

    majority_class = np.unique(np.asarray(test.label.values), return_counts = True)
   
    descriptions["train"]["label_distribution"] = {str(k):v for k, v in dict(np.asarray(majority_class).T).items()}
    descriptions["train"]["majority_class"] =  round(majority_class[-1].max() / majority_class[-1].sum() * 100,2)

    if args["query"]:

        descriptions["train"]["ave_query_length"] =  math.ceil(np.asarray([len(x.split()) for x in train["query"].values]).mean())
        descriptions["dev"]["ave_query_length"] =  math.ceil(np.asarray([len(x.split()) for x in dev["query"].values]).mean())
        descriptions["test"]["ave_query_length"] =  math.ceil(np.asarray([len(x.split()) for x in test["query"].values]).mean())
    
    return descriptions


def encode_it(tokenizer, max_length, *arguments):

    """
    returns token type ids, padded doc and 
    """

    if len(arguments) > 1:

        dic = tokenizer.encode_plus(arguments[0], arguments[1],
                                        add_special_tokens = True,
                                        max_length = max_length,
                                        padding = 'max_length',
                                        return_token_type_ids = True,
                                        truncation = True)

    else:
  
        dic = tokenizer.encode_plus(arguments[0],
                                        add_special_tokens = True,
                                        max_length = max_length,
                                        padding = 'max_length',
                                        return_token_type_ids = True,
                                        truncation = True)
       
    return dic


import torch

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
