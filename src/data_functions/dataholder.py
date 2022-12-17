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

        ## if loading rationales we have to also include the importance metric

        if for_rationale:
            
            path += args["importance_metric"] + "-"

        train = pd.read_csv(path + "train.csv").to_dict("records")#[1066:1260] #list of dic
        dev = pd.read_csv(path + "dev.csv").to_dict("records")#[1066:1160] # for testing by cass
        test = pd.read_csv(path + "test.csv").to_dict("records")#[1006:1020] # for testing by cass
        # print(' ======= test set ---- one data')
        # print(len(test))
        # print(test[0])
        

        print("*** loading data in dataholder")

        ## if we are dealing with a query we need to account for the query length as well
        if args.query:
            
            max_len = round(max([len(x["document"].split()) for x in train])) + \
                        max([len(x["query"].split()) for x in train])
            max_len = round(max_len)

        else:
            
            max_len = round(max([len(x["text"].split()) for x in train]))


        max_len = min(max_len, 512)
        self.max_len = max_len

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

        print("*** dataholder ready")

    def as_dataframes_(self):

        return self.return_as_frames 




# class BERT_HOLDER_interpolation():
#     """
#     class that holds our data, pretrained tokenizer and set sequence length 
#     for a classification task
#     """
#     def __init__(self, path = str, b_size = 8 , 
#                 for_rationale = False, 
#                 stage = "train", return_as_frames = False, interpolation=True, importance_scores=None, M_set = 0):
  
#         """
#         loads data for a classification task from preprocessed .csv 
#         files in the dataset/data folder
#         and returns three dataholders : train, dev, test
#         """

#         assert type(b_size) == int
    
#         self.batch_size = b_size

#         if for_rationale:
            
#             if args.use_tasc: args["importance_metric"] = "tasc_" + args["importance_metric"]
#             path += args["thresholder"] + "/" + args["importance_metric"] + "-"

#         with open(f"{path}train.json", "r") as file: train = json.load(file)[111:132]
#         with open(f"{path}dev.json", "r") as file: dev = json.load(file)[111:132]
#         with open(f"{path}test.json", "r") as file: test = json.load(file)[222:232]

#         ## load data
#         if interpolation == True: 
#             test = sample(test, 50)
#             print(' for interpolation !!!!!!!111  the test len is ', len(test))
#         else:
#             pass


#         print("*** loading data in dataholder")

#         ## if we are dealing with a query we need to account for the query length as well
#         if args.query:
            
#             max_len = round(max([len(x["document"].split()) for x in train])) + \
#                         max([len(x["query"].split()) for x in train])
#             max_len = round(max_len)

#         else:
#             max_len = round(max([len(x["text"].split()) for x in train]))

#         if args['dataset'] == 'SST':
#             max_len = 48

#         self.max_len = min(max_len, 256)
#         print(' ------ max_len', max_len)
#         # load the pretrained tokenizer
#         pretrained_weights = args.model
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, local_files_only=False)

#         self.nu_of_labels = len(np.unique([x["label"] for x in train]))
#         self.vocab_size = len(self.tokenizer)

#         if args.query:
            
#             train = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["document"], dic["query"]) for dic in train]
#             dev = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["document"], dic["query"]) for dic in dev]
#             test_S4 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in test.copy()]
#             test_S3 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in test.copy()]
#             test_S2 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in test.copy()]
#             test_S1 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in test.copy()]

#         else:

#             train = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in train]
#             dev = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in dev]
#             cpy_list = []
#             for li in test:
#                 d2 = copy.deepcopy(li)
#                 cpy_list.append(d2)
#             test_S4 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in cpy_list]

#             cpy_list = []
#             for li in test:
#                 d2 = copy.deepcopy(li)
#                 cpy_list.append(d2)
#             test_S3 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in cpy_list]

#             cpy_list = []
#             for li in test:
#                 d2 = copy.deepcopy(li)
#                 cpy_list.append(d2)
#             test_S2 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in cpy_list]

#             cpy_list = []
#             for li in test:
#                 d2 = copy.deepcopy(li)
#                 cpy_list.append(d2)
#             test_S1 = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in cpy_list]


#         importance_scores = importance_scores

#         # print(importance_scores.get('test_358').get('attention')) # dict_keys(['random', 'attention', 'gradients', 'ig', 'scaled attention', 'deeplift', 'gradientshap'])
#         # x = np.argsort(importance_scores.get('test_358').get('attention'))[::-1][:4]
#         # print("Indices:",x)


#         for i, test_single_data in enumerate(test_S4): # together 50 test 
#             annotation_id = test_single_data['annotation_id']
#             this_token_attention_importance = np.argsort(importance_scores.get(annotation_id).get('attention'))[::-1][:4].copy()
#             test_S4[i]['input_ids'][0][this_token_attention_importance] = test_single_data['input_ids'][0][this_token_attention_importance] + random.randint(1, 9999) 
        
#         for i, test_single_data in enumerate(test_S3): # together 50 test 
#             annotation_id = test_single_data['annotation_id']
#             this_token_attention_importance = np.argsort(importance_scores.get(annotation_id).get('attention'))[::-1][:4][-3:].copy()
#             test_S3[i]['input_ids'][0][this_token_attention_importance] = test_single_data['input_ids'][0][this_token_attention_importance] + random.randint(1, 9999) 
        
#         for i, test_single_data in enumerate(test_S2): # together 50 test 
#             annotation_id = test_single_data['annotation_id']
#             this_token_attention_importance = np.argsort(importance_scores.get(annotation_id).get('attention'))[::-1][:4][-2:].copy()
#             test_S2[i]['input_ids'][0][this_token_attention_importance] = test_single_data['input_ids'][0][this_token_attention_importance] + random.randint(1, 9999) 
        
#         for i, test_single_data in enumerate(test_S1): # together 50 test 
#             annotation_id = test_single_data['annotation_id']
#             this_token_attention_importance = np.argsort(importance_scores.get(annotation_id).get('attention'))[::-1][:4][-1:].copy()
#             test_S1[i]['input_ids'][0][this_token_attention_importance] = test_single_data['input_ids'][0][this_token_attention_importance] + random.randint(1, 9999) 
        


#         shuffle_during_iter = True

#         if stage != "train": 

#             # sort by length for evaluation and rationale extraction
#             train = sorted(train, key = lambda x : x["lengths"], reverse = False)
#             dev = sorted(dev, key = lambda x : x["lengths"], reverse = False)
#             test_S1 = sorted(test_S1, key = lambda x : x["lengths"], reverse = False)
#             test_S2 = sorted(test_S2, key = lambda x : x["lengths"], reverse = False)
#             test_S3 = sorted(test_S3, key = lambda x : x["lengths"], reverse = False)
#             test_S4 = sorted(test_S4, key = lambda x : x["lengths"], reverse = False)

#             shuffle_during_iter = False
        
#         if return_as_frames:

#             self.return_as_frames = {
#                 "train" : pd.DataFrame(train),
#                 "dev" : pd.DataFrame(dev),
#                 "test_S1" : pd.DataFrame(test_S1),
#                 "test_S2" : pd.DataFrame(test_S2),
#                 "test_S3" : pd.DataFrame(test_S3),
#                 "test_S4" : pd.DataFrame(test_S4),
#             }

#         # prepare data-loaders for training
#         self.train_loader = DataLoader(
#             train,
#             batch_size = self.batch_size,
#             shuffle = shuffle_during_iter,
#             pin_memory = False,
#             #collate_fn=lambda x: x, # for not "equal size" batch issue # debug by cass
#         )

#         self.dev_loader = DataLoader(
#             dev,
#             batch_size = self.batch_size,
#             shuffle = shuffle_during_iter,
#             pin_memory = False,
#             #collate_fn=lambda x: x, # for not "equal size" batch issue
#         )


#         self.test_loader_S4 = DataLoader(  # the S4
#             test_S4,
#             batch_size = self.batch_size,
#             shuffle = shuffle_during_iter,
#             pin_memory = False,
#             #collate_fn=lambda x: x, # for not "equal size" batch issue
#         )

#         self.test_loader_S3 = DataLoader(  # the S4
#             test_S3,
#             batch_size = self.batch_size,
#             shuffle = shuffle_during_iter,
#             pin_memory = False,
#             #collate_fn=lambda x: x, # for not "equal size" batch issue
#         ) 

#         self.test_loader_S2 = DataLoader(  # the S4
#             test_S2,
#             batch_size = self.batch_size,
#             shuffle = shuffle_during_iter,
#             pin_memory = False,
#             #collate_fn=lambda x: x, # for not "equal size" batch issue
#         )

#         self.test_loader_S1 = DataLoader(  # the S4
#             test_S1,
#             batch_size = self.batch_size,
#             shuffle = shuffle_during_iter,
#             pin_memory = False,
#             #collate_fn=lambda x: x, # for not "equal size" batch issue
#         ) 
#         print("*** dataholder ready")

#     def as_dataframes_(self):

#         return self.return_as_frames 

### copy from ood 
'''

from src.data_functions.non_transformer import pretrained_embeds, extract_vocabulary_
import os
## holds our data
class KUMA_RL_HOLDER():
    
    """
    Data holder for our inherenlty faithful models
    RL + KUMA    
    """

    def __init__(self, path : str, b_size : int =  8 , 
                ood : bool = False, ood_dataset_ : int = 0):
  
        assert type(b_size) == int
    
        self.batch_size = b_size

        ## load data
        with open(f"{path}train.json", "r") as file: train = json.load(file)#[:32]
        with open(f"{path}dev.json", "r") as file: dev = json.load(file)#[:32]
        with open(f"{path}test.json", "r") as file: test = json.load(file)#[:32]

        print("*** loading data in dataholder")
            
        self.max_len = round(max([len(x["text"].split()) for x in train]))
        self.nu_of_labels = len(np.unique([x["label"] for x in train]))

        ## check if we processed already the pretrained embeds and vocab
        vocab_fname = os.path.join(
            args.data_dir,
            "vocabulary.json"
        )

        if os.path.exists(vocab_fname):

            with open(vocab_fname, "r") as f: self.w2ix = json.load(f)

        else:

            self.w2ix = extract_vocabulary_(
                data = train
            )

            with open(vocab_fname, "w") as f: 
                json.dump(
                    {k:int(v) for k,v in self.w2ix.items()},
                    f
                )

        self.vocab_size = len(self.w2ix)

        embed_fname = os.path.join(
            args.data_dir,
            f"{args.embed_model}_embeds.npy"
        )
        if os.path.exists(embed_fname):

            pass

        else: ## if not lets create them and save them

            ix2w = {v:k for k,v in self.w2ix.items()}

            embeds = pretrained_embeds(
                model = args.embed_model,
                ix_to_word=ix2w
            ).processed()

            np.save(
                embed_fname,
                embeds
            )

        train = self._process_data_(train) ## 
        dev = self._process_data_(dev)
        test = self._process_data_(test)

        # prepare data-loaders for training
        self.train_loader = DataLoader(
            train,
            batch_size = self.batch_size,
            shuffle = True,
            pin_memory = False
        )

        self.dev_loader = DataLoader(
            dev,
            batch_size = self.batch_size,
            pin_memory = False
        )

        self.test_loader = DataLoader(
            test,
            batch_size = self.batch_size,
            pin_memory = False
        )  

    def _process_data_(self, data_to_process):
        
        return [self._process_instance_(x) for x in data_to_process]
        

    def _process_instance_(self, instance):
        
        instance["input_ids"] = [self.w2ix["<SOS>"]] + \
            [self.w2ix[w] if w in self.w2ix else self.w2ix["<UNKN>"] for w in instance["text"].split(" ")] + \
                [self.w2ix["<EOS>"]]
        
        # del instance["text"]
        instance["length"] = len(instance["input_ids"])
        instance["input_ids"] = np.asarray(
            self._pad_data_(
                tokenized_ids = instance["input_ids"], 
                pad_length = self.max_len
            )
        )

        if instance["input_ids"][-1] != 0:
            
            instance["input_ids"][-1] = self.w2ix["<EOS>"]

        return instance

    def _pad_data_(self, tokenized_ids : list, pad_length : int):
        
        """
        args:
            tokenized_text - tensor to pad
            pad_length - the size to pad to

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """

        diff = pad_length - len(tokenized_ids)

        padded = tokenized_ids + [0]*diff
        
        return padded[:pad_length]

'''