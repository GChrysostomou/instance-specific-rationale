from src.common_code.useful_functions import encode_it
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class classification_dataholder():
    """
    class that holds our data, pretrained tokenizer and set sequence length 
    for a classification task
    """
    def __init__(self, path = str, b_size = 8 , mask_list = list, 
                for_rationale = False, variable = False, return_as_frames = False):
        
        assert type(b_size) == int
    
        self.batch_size = b_size


        """
        loads data for a classification task from preprocessed .csv 
        files in the dataset/data folder
        and returns three dataholders : train, dev, test
        """

        ## if loading rationales we have to also include the importance metric

        if for_rationale:
            
            path += args["importance_metric"] + "-"

        train = pd.read_csv(path + "train.csv")#.sample(frac = 0.01, random_state = 12)
        dev = pd.read_csv(path + "dev.csv")#.sample(frac = 0.01, random_state = 12)
        test = pd.read_csv(path + "test.csv")#.sample(frac = 0.01, random_state = 12)

        ## if we are dealing with a query we need to account for the query length as well
        if args.query:
            
            max_len = round(max([len(x.split()) for x in train.document.values])) + \
                        max([len(x.split()) for x in train["query"].values])
            max_len = round(max_len)

        else:
            
            max_len = round(max([len(x.split()) for x in train.text.values]))

        max_len = min(max_len, 512)

        # load the pretrained tokenizer
        pretrained_weights = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

        self.nu_of_labels = len(train.label.unique())

        ### I use pandas to preprocess everything just to make sure everything is in place
        ## I am aware it is a bit slower but we are not dealing with massive data
        if args.query:
            
            train["text"] = train.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["document"], x["query"]), axis = 1)
            dev["text"] = dev.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["document"], x["query"]), axis = 1)
            test["text"] = test.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["document"], x["query"]), axis = 1)

            # used only on rationale extraction
            train["query_mask"] = train.text.transform(lambda x: list((np.asarray(x["token_type_ids"]) == 0).astype(int)))
            dev["query_mask"] = dev.text.transform(lambda x: list((np.asarray(x["token_type_ids"]) == 0).astype(int)))
            test["query_mask"] = test.text.transform(lambda x: list((np.asarray(x["token_type_ids"]) == 0).astype(int)))

        else:

            train["text"] = train.apply(lambda x: encode_it(self.tokenizer, 
                            max_len,  x["text"]), axis = 1)
            dev["text"] = dev.apply(lambda x: encode_it(self.tokenizer, 
                            max_len,  x["text"]), axis = 1)
            test["text"] = test.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["text"]), axis = 1)

            # used only on rationale extraction
            train["query_mask"] = train.text.transform(lambda x:list((np.asarray(x["input_ids"]) != 0).astype(int)) )
            dev["query_mask"] = dev.text.transform(lambda x:list((np.asarray(x["input_ids"]) != 0).astype(int)))
            test["query_mask"] = test.text.transform(lambda x:list((np.asarray(x["input_ids"]) != 0).astype(int)))

        
        train["token_type_ids"] = train.text.transform(lambda x:x["token_type_ids"])
        dev["token_type_ids"] = dev.text.transform(lambda x:x["token_type_ids"])
        test["token_type_ids"] = test.text.transform(lambda x:x["token_type_ids"])


        train["attention_mask"] = train.text.transform(lambda x:x["attention_mask"])
        train["text"] = train.text.transform(lambda x:x["input_ids"])
        train["lengths"] = train.attention_mask.apply(lambda x: sum(x))

        
        dev["attention_mask"] = dev.text.transform(lambda x:x["attention_mask"])
        dev["text"] = dev.text.transform(lambda x:x["input_ids"])
        dev["lengths"] = dev.attention_mask.apply(lambda x: sum(x))

        test["attention_mask"] = test.text.transform(lambda x:x["attention_mask"])
        test["text"] = test.text.transform(lambda x:x["input_ids"])
        test["lengths"] = test.attention_mask.apply(lambda x: sum(x))

        # sort by length 
        train = train.sort_values("lengths", ascending = True)#### default True
        dev = dev.sort_values("lengths", ascending = True)
        test = test.sort_values("lengths", ascending = True)
        
        if return_as_frames:

            self.return_as_frames = {
                "train" : train,
                "dev" : dev,
                "test" : test
            }

        # prepare data-loaders for training
        columns = ["text", "lengths", "label", "annotation_id", "query_mask", 
                    "token_type_ids", "attention_mask"]

        self.train_loader = DataLoader(train[columns].values.tolist(),
                                batch_size = self.batch_size,
                                shuffle = False,
                                pin_memory = False)

        self.dev_loader = DataLoader(dev[columns].values.tolist(),
                                batch_size = self.batch_size,
                                shuffle = False,
                                pin_memory = False)

        self.test_loader = DataLoader(test[columns].values.tolist(),
                                batch_size = self.batch_size,
                                shuffle = False,
                                pin_memory = False)  


    def as_dataframes_(self):

        return self.return_as_frames