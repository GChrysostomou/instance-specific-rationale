# """
# contains functions for helping with the loading, processing, description and preparation of the datasets
# """

import pandas as pd
import numpy as np
import math
import json 
import torch
import gc

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def contigious_(importance_scores, tokens_to_mask):

    ngram = torch.stack([importance_scores[i:i + tokens_to_mask] for i in range(len(importance_scores) - tokens_to_mask + 1)])
    indxs = [torch.arange(i, i+tokens_to_mask) for i in range(len(importance_scores) - tokens_to_mask + 1)]
    top_k = indxs[ngram.sum(-1).argmax()]

    return top_k

def topk_(importance_scores, tokens_to_mask):

    top_k = torch.topk(importance_scores, tokens_to_mask).indices

    return top_k


def batch_from_dict_(batch_data, metadata, target_key = "original prediction"):
    new_tensor = []
    for _id_ in batch_data["annotation_id"]:
        # "annotatiion_id" "input_ids" "lengths" "labels" "token_type_ides" "attention_mask" "query_mask" "special_tokens"
        new_tensor.append(
            metadata[_id_][target_key]
        )
    return torch.tensor(new_tensor)#.to(device)


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
    print(tokens)
    print(len(tokens))

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



def mask_topk(sentences, scores, length_to_mask):

    length_to_mask = torch.topk(scores, length_to_mask) #.to(device)
    mask = torch.ones(sentences.shape)
    mask = mask.scatter_(-1,  length_to_mask[1], 0)

    return sentences * mask.long().to(device)

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


# ########################## RATIONALE MASKS ---> CORE 
def create_rationale_mask_(
        
        importance_scores: torch.tensor, 
        no_of_masked_tokens: np.ndarray,
        special_tokens: dict,
        method = "topk", batch_input_ids = None
    ):

    rationale_mask = []

    for _i_ in range(importance_scores.size(0)):
        
        score = importance_scores[_i_]
        tokens_to_mask = int(no_of_masked_tokens[_i_])
        
        ## if contigious or not a unigram (unigram == topk of 1)
        if method == "contigious" and tokens_to_mask > 1:

            top_k = contigious_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )
        
        else:

            top_k = topk_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )

        ## create the instance specific mask
        ## 1 represents the rationale :)
        ## 0 represents tokens that we dont care about :'(
        mask = torch.zeros(score.shape).to(device)
        mask = mask.scatter_(-1,  top_k.to(device), 1).long()

        ## now if we have a query we need to preserve the query in the mask
        if batch_input_ids is not None:
            
            sos_eos = torch.where( batch_input_ids[_i_] == special_tokens["sep_token_id"][0].item())[0]
            seq_length = sos_eos[0]
            query_end = sos_eos[1]

            mask[seq_length: query_end+1] = 1 

        rationale_mask.append(mask)

    rationale_mask = torch.stack(rationale_mask).to(device)

    return rationale_mask

## used for preserving queries
def create_only_query_mask_(batch_input_ids : torch.tensor, special_tokens : dict):
    query_mask = []

    for seq in batch_input_ids:
        
        only_query_mask = torch.zeros(seq.shape).to(device)
        sos_eos = torch.where(seq == special_tokens["sep_token_id"][0].item())[0]

        seq_length = sos_eos[-2] # query start
        query_end = sos_eos[-1]
        
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



# nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])


def describe_data_stats(path_to_data, path_to_stats):
    """ 
    returns dataset statistics such as : 
                                        - number of documens
                                        - average sequence length
                                        - average query length (if QA)
    """

    descriptions = {"train":{}, "dev":{}, "test":{}}
    
    for split_name in descriptions.keys():

        data = pd.read_csv(f"{path_to_data}{split_name}.csv").to_dict("records")

        if "query" in data[0].keys(): 


            avg_ctx_len = np.asarray([len(x["document"].split(" ")) for x in data]).mean()
            avg_query_len = np.asarray([len(x["query"].split(" ")) for x in data]).mean()

            descriptions[split_name]["avg. context length"] = int(avg_ctx_len)
            descriptions[split_name]["avg. query length"] = int(avg_query_len)

        else:

            avg_seq_len = np.asarray([len(x["text"].split(" ")) for x in data]).mean()

            descriptions[split_name]["avg. sequence length"] = int(avg_seq_len)

        descriptions[split_name]["no. of documents"] = int(len(data))
        
        label_nos = np.unique(np.asarray([x["label"] for x in data]), return_counts = True)

        for label, no_of_docs in zip(label_nos[0], label_nos[1]):

            descriptions[split_name][f"docs in label-{label}"] = int(no_of_docs)
    
    ## save descriptors
    fname = path_to_stats + "dataset_statistics.json"

    with open(fname, 'w') as file:
        
            json.dump(
                descriptions,
                file,
                indent = 4
            ) 


    del data
    del descriptions
    gc.collect()

    return

def encode_plusplus_(data_dict, tokenizer, max_length, *arguments):

    """
    returns token type ids, padded doc and 
    """

    ## if len(args)  > 1 that means that we have ctx + query
    if len(arguments) > 1:

        model_inputs = tokenizer.encode_plus(
            arguments[0], 
            arguments[1],
            add_special_tokens = True,
            max_length = max_length,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True,
            return_tensors = "pt"             
        )



        data_dict.update(model_inputs)

        del data_dict["document"]
        del data_dict["query"]

        ## query mask used_only for rationale extraction and for masking importance metrics
        ## i.e. keeping only the contxt not the query
        init_mask_ = torch.where(model_inputs["input_ids"] == tokenizer.sep_token_id)[1][0]
        fin_mask = model_inputs["input_ids"].size(-1)
        range_to_zero = torch.arange(init_mask_, fin_mask)
        model_inputs["query_mask"] = model_inputs["attention_mask"].clone()
        model_inputs["query_mask"].squeeze(0)[range_to_zero] = 0
        ## preserve cls token
        model_inputs["query_mask"].squeeze(0)[0] = 0
        

    else:
  
        model_inputs = tokenizer.encode_plus(
            arguments[0], 
            add_special_tokens = True,
            max_length = max_length,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True,
            return_tensors = "pt"             
        )



        del data_dict["text"]
    
        init_mask_ = torch.where(model_inputs["input_ids"] == tokenizer.sep_token_id)[1][0]
        model_inputs["query_mask"] = model_inputs["attention_mask"].clone()
        ## preserve cls token
        model_inputs["query_mask"].squeeze(0)[0] = 0

    ## context length
    model_inputs["lengths"] = init_mask_
    model_inputs["special tokens"] = {
        "pad_token_id" : tokenizer.pad_token_id,
        "sep_token_id" : tokenizer.sep_token_id
    }

    data_dict.update(model_inputs)

    return data_dict