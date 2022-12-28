from src.data_functions.useful_functions import encode_plusplus_
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import re
from random import sample

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class BERT_HOLDER():
    """
    class that holds our data, pretrained tokenizer and set sequence length 
    for a classification task
    """
    def __init__(self, path = str, b_size = 8 , #mask_list = list, 
                for_rationale = False, variable = False, 
                return_as_frames = False, stage = "train",
                ):
        
        assert type(b_size) == int
    
        self.batch_size = b_size
        """
        loads data for a classification task from preprocessed .csv =
        files in the dataset/data folder
        and returns three dataholders : train, dev, test
        """

        # ## if loading rationales we have to also include the importance metric

        # if for_rationale:
            
        #     path += args["importance_metric"] + "-"

        train = pd.read_csv(path + "train.csv").to_dict("records")#[1066:1260] #list of dic
        dev = pd.read_csv(path + "dev.csv").to_dict("records")#[1066:1160] # for testing by cass
        test = pd.read_csv(path + "test.csv").to_dict("records")
        ## if we are dealing with a query we need to account for the query length as well

        if args.query:
            # max_len = round(max([len(x["document"].split()) for x in test])) + \
            #                 max([len(x["query"].split()) for x in test])

            max_len = round(max([len(x["document"].split()) for x in train])) + \
            max([len(x["query"].split()) for x in train])
            max_len = round(max_len)
            print('        the document max_len (in train):', round(max([len(x["document"].split()) for x in train])))
            print('        the query max_len (in train):', round(max([len(x["query"].split()) for x in train])))
            print('        the query mean len (in train):', sum([len(x["query"].split()) for x in train])/len(train))

        else:
            
            max_len = round(max([len(x["text"].split()) for x in train]))

        print('        the task max_len (in train):', max_len)

        max_len = min(max_len, 512)
        self.max_len = max_len # 还在inti 里面

        # load the pretrained tokenizer
        pretrained_weights = args.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights) # by cass ood time dataholders.py (, local_files_only=True)
        self.nu_of_labels = len(np.unique([x["label"] for x in train]))
        #print('self.nu_of_labels  ', self.nu_of_labels)


        if args.query:
            train = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in dev]
            test = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in test]

        else:

            train = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in dev]
            test= [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in test]

        shuffle_during_iter = True


        # IF TESTING LOCALLY
        test = test[:200]


        

        if stage != "train": 

            # ###### sort by length for evaluation and rationale extraction
            train = sorted(train, key = lambda x : x["lengths"], reverse = False)
            dev = sorted(dev, key = lambda x : x["lengths"], reverse = False)
            test = sorted(test, key = lambda x : x["lengths"], reverse = True)


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
            pin_memory = False,
        )

        self.dev_loader = DataLoader(
            dev,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.test_loader = DataLoader(
            test,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )


    def as_dataframes_(self):

        return self.return_as_frames 


from random_word import RandomWords
r = RandomWords()

# Return a single random word
def add_random_word(dataset, fixed_set):
    text = []
    for single_text in dataset['text']:

        if fixed_set == "fixed0":
            p = r.get_random_word() + " " + r.get_random_word() + " " + r.get_random_word() + " " + r.get_random_word() + r.get_random_word() + r.get_random_word()# fixed 0 --> 4 random  --> add 3
        elif fixed_set == "fixed1":
            p = single_text + " " + r.get_random_word() + " " + r.get_random_word() + " " + r.get_random_word() + r.get_random_word() + r.get_random_word()# fixed 3 --> three random
        elif fixed_set == "fixed2":
            p = single_text + " " + r.get_random_word() + " " + r.get_random_word() + r.get_random_word() + r.get_random_word()
        elif fixed_set == "fixed3":
            p = single_text + " " + r.get_random_word() + r.get_random_word() + r.get_random_word()
        elif fixed_set == "fixed4":
            p = single_text + " " + r.get_random_word() + r.get_random_word() 
        elif fixed_set == "fixed5":
            p = single_text + " " + r.get_random_word() 
        else:
            pass
        text.append(p)
    dataset["text"] = text
    return dataset

class BERT_HOLDER_interpolation():
    """
    class that holds our data, pretrained tokenizer and set sequence length 
    for a classification task
    """
    def __init__(self, path = str, b_size = 8 , FA_name = "attention",
                for_rationale = False, variable = False, 
                return_as_frames = False, stage = "interpolation",
                ):
        
        assert type(b_size) == int
    
        self.batch_size = b_size
        """
        loads data for a classification task from preprocessed .csv =
        files in the dataset/data folder
        and returns three dataholders : train, dev, test
        """

        # ## if loading rationales we have to also include the importance metric

        # if for_rationale:
            
        #     path += args["importance_metric"] + "-"
        fixed1 = pd.read_csv(f"./extracted_rationales/sst/data/fixed1/{FA_name}-test.csv")
        fixed2 = pd.read_csv(f"./extracted_rationales/sst/data/fixed2/{FA_name}-test.csv")
        fixed3 = pd.read_csv(f"./extracted_rationales/sst/data/fixed3/{FA_name}-test.csv")
        fixed4 = pd.read_csv(f"./extracted_rationales/sst/data/fixed4/{FA_name}-test.csv")
        fixed5 = pd.read_csv(f"./extracted_rationales/sst/data/fixed5/{FA_name}-test.csv")
        fixed6 = pd.read_csv(f"./extracted_rationales/sst/data/fixed6/{FA_name}-test.csv")

        # print('  ')
        # print('  ')
        # print('  ')
        # print(' BEFORE FILTER OUT CLS ANS SEP DATA ', len(fixed6))
        fixed6 = fixed6[fixed6["text"].str.contains("[CLS]")==False]
        fixed6 = fixed6[fixed6["text"].str.contains("[SEP]")==False]
        #print(' AFTER FILTER OUT CLS ANS SEP DATA ', len(fixed6))
        fixed6 = fixed6.sample(50)
        
        fixed6.to_csv(f"./interpolation/sst/{FA_name}-fixed6-AnalysisSamples.csv")
        ids = fixed6["annotation_id"]
        

        fixed1 = fixed1.loc[fixed1['annotation_id'].isin(ids)]
        fixed2 = fixed2.loc[fixed2['annotation_id'].isin(ids)]
        fixed3 = fixed3.loc[fixed3['annotation_id'].isin(ids)]
        fixed4 = fixed4.loc[fixed4['annotation_id'].isin(ids)]
        fixed5 = fixed5.loc[fixed5['annotation_id'].isin(ids)]
        fixed0 = fixed1.copy(deep=True)
        #print(' ================> fixed2 len :', len(fixed2))
        

        fixed0 = add_random_word(fixed0, "fixed0").to_dict("records")
        fixed1 = add_random_word(fixed1, "fixed1").to_dict("records")
        fixed2 = add_random_word(fixed2, "fixed2").to_dict("records")        
        fixed3 = add_random_word(fixed3, "fixed3").to_dict("records")
        fixed4 = add_random_word(fixed4, "fixed4").to_dict("records")
        fixed5 = add_random_word(fixed5, "fixed5").to_dict("records")
        
        fixed6 = fixed6.to_dict("records") # no need to add



        ## if we are dealing with a query we need to account for the query length as well
        if args.query:
            max_len = round(max([len(x["document"].split()) for x in fixed6])) + \
                        max([len(x["query"].split()) for x in fixed6])
            max_len = round(max_len)

        else:
            max_len = round(max([len(x["text"].split()) for x in fixed6]))


        max_len = min(max_len, 512)
        self.max_len = max_len

        # load the pretrained tokenizer
        pretrained_weights = args.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights) # by cass ood time dataholders.py (, local_files_only=True)

        self.nu_of_labels = len(np.unique([x["label"] for x in fixed4]))
        #print('self.nu_of_labels  ', self.nu_of_labels)


        if args.query:

            fixed6 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed6]
            fixed5 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed5]
            fixed4 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed4]
            fixed3 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed3]
            fixed2 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed2]
            fixed1 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed1]
            fixed0 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in fixed0]
        else:
            fixed6 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed6]
            fixed5 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed5]
            fixed4 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed4]
            fixed3 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed3]
            fixed2 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed2]
            fixed1 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed1]
            fixed0 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed0]


        fixed6 = sorted(fixed6, key = lambda x : x["lengths"], reverse = False)
        fixed5 = sorted(fixed5, key = lambda x : x["lengths"], reverse = False)
        fixed4 = sorted(fixed4, key = lambda x : x["lengths"], reverse = False)
        fixed3 = sorted(fixed3, key = lambda x : x["lengths"], reverse = False)
        fixed2 = sorted(fixed2, key = lambda x : x["lengths"], reverse = False)
        fixed1 = sorted(fixed1, key = lambda x : x["lengths"], reverse = False)
        fixed0 = sorted(fixed0, key = lambda x : x["lengths"], reverse = False)
        shuffle_during_iter = False
        
        if return_as_frames:

            self.return_as_frames = {
                "fixed6" : pd.DataFrame(fixed6),
                "fixed5" : pd.DataFrame(fixed5),
                "fixed4" : pd.DataFrame(fixed4),
                "fixed3" : pd.DataFrame(fixed3),
                "fixed2" : pd.DataFrame(fixed2),
                "fixed1" : pd.DataFrame(fixed1),
                "fixed0" : pd.DataFrame(fixed0),
            }

        # prepare data-loaders for training
        self.fixed6_loader = DataLoader(
            fixed6,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.fixed5_loader = DataLoader(
            fixed5,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.fixed4_loader = DataLoader(
            fixed4,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.fixed3_loader = DataLoader(
            fixed3,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.fixed2_loader = DataLoader(
            fixed2,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.fixed1_loader = DataLoader(
            fixed1,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        self.fixed0_loader = DataLoader(
            fixed0,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False,
        )

        print("*** dataholder ready")

    def as_dataframes_(self):

        return self.return_as_frames 


