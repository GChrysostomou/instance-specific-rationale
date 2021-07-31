from src.data_functions.useful_functions import encode_plusplus_
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
                for_rationale = False, variable = False, 
                return_as_frames = False, stage = "train"):
        
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

        train = pd.read_csv(path + "train.csv").to_dict("records")#[:32]
        dev = pd.read_csv(path + "dev.csv").to_dict("records")#[:32]
        test = pd.read_csv(path + "test.csv").to_dict("records")#[:32]

        print("*** loading data in dataholder")

        ## if we are dealing with a query we need to account for the query length as well
        if args.query:
            
            max_len = round(max([len(x["document"].split()) for x in train])) + \
                        max([len(x["query"].split()) for x in train])
            max_len = round(max_len)

        else:
            
            max_len = round(max([len(x["text"].split()) for x in train]))

        max_len = min(max_len, 256)

        # load the pretrained tokenizer
        pretrained_weights = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

        self.nu_of_labels = len(np.unique([x["label"] for x in train]))

        if args.query:
            
            train = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in dev]
            test = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in test]

        else:

            train = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in dev]
            test= [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in test]

        shuffle_during_iter = True

        if stage != "train": 

            # sort by length for evaluation and rationale extraction
            train = sorted(train, key = lambda x : x["lengths"], reverse = False)
            dev = sorted(dev, key = lambda x : x["lengths"], reverse = False)
            test = sorted(test, key = lambda x : x["lengths"], reverse = False)

            shuffle_during_iter = False
        
        if return_as_frames:

            self.return_as_frames = {
                "train" : pd.DataFrame(train),
                "dev" : pd.DataFrame(dev),
                "test" : pd.DataFrame(test)
            }

        # prepare data-loaders for training
        self.train_loader = DataLoader(
            train,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False
        )

        self.dev_loader = DataLoader(
            dev,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False
        )

        self.test_loader = DataLoader(
            test,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False
        )  

        print("*** dataholder ready")

    def as_dataframes_(self):

        return self.return_as_frames 

       
    def as_dataframes_(self):

        return self.return_as_frames