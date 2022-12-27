import spacy
import torch
import numpy as np
import pandas as pd
import json
import gc

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])


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
        #print(arguments[0], '/////////////////////////////////////',arguments[1])

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

    

        # print(tokenizer)

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
