from src.data_functions.useful_functions import encode_plusplus_, encode_plusplus_t5
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5TokenizerFast, T5Tokenizer
import json
import re
from random import sample
import transformers
transformers.logging.set_verbosity_error()
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

        train = pd.read_csv(path + "train.csv").to_dict("records")#[1166:1170] #list of dic
        dev = pd.read_csv(path + "dev.csv").to_dict("records")#[1166:1170] # for testing by cass
        test = pd.read_csv(path + "test.csv").to_dict("records")#[1166:1170]
        ## if we are dealing with a query we need to account for the query length as well
        print(train[:3])

        if args.query:
            max_len = round(max([len(str(x["document"]).split()) for x in train])) + \
            max([len(str(x["query"]).split()) for x in train])
            max_len = round(max_len)
            

        else:
            max_len = round(max([len(x["text"].split()) for x in train]))
            

        if 'csl' in args.dataset:
            max_len = 256
        elif 'ChnSentiCorp' in args.dataset:
            max_len = 128
        elif 'ant' in args.dataset:
            max_len = 128
        else: pass
        

        max_len = min(max_len, 512)
        print('        the task max_len (in train):', max_len)
        self.max_len = max_len # 还在inti 里面

        # load the pretrained tokenizer
        pretrained_weights = args.model
        print(' --------------- token model -->', pretrained_weights)
      
        
        if "flaubert" in args['model_abbreviation']:
            from transformers import FlaubertTokenizer
            self.tokenizer = FlaubertTokenizer.from_pretrained(pretrained_weights, local_files_only=False) 
        else: self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, local_files_only=False) # by cass ood time dataholders.py (, local_files_only=True)
        self.nu_of_labels = len(np.unique([x["label"] for x in train]))


        if args.query:
            if 'paws' in args['dataset']:
                train = [encode_plusplus_(dic, self.tokenizer, max_len,  str(dic["document"]), str(dic["query"])) for dic in train]
            else: train = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in dev]
            test = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["document"], dic["query"]) for dic in test]

        else:
            train = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in dev]
            test= [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in test]

        shuffle_during_iter = True

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




class mT5_HOLDER():
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

        train = pd.read_csv(path + "train.csv").to_dict("records")#[1166:1200] #list of dic
        dev = pd.read_csv(path + "dev.csv").to_dict("records")#[1:11] # for testing by cass
        test = pd.read_csv(path + "test.csv").to_dict("records")#[1:11]
        ## if we are dealing with a query we need to account for the query length as well
        print(train[:1])

        if args.query:
            max_len = round(max([len(str(x["document"]).split()) for x in train])) + \
            max([len(str(x["query"]).split()) for x in train])
            max_len = round(max_len)
        else:
            max_len = round(max([len(x["text"].split()) for x in train]))
 

        max_len = min(max_len, 512)
        print('        the task max_len (in train):', max_len)
        self.max_len = max_len # 还在inti 里面
        unique_labels_num_format = np.unique([x["label"] for x in train])
        self.nu_of_labels = len(unique_labels_num_format)
        print('  self.nu_of_labels', self.nu_of_labels)
        # unique_labels = str(unique_labels)[1:-1]
        # print(f"==>> unique_labels: {unique_labels}")

        
        
        # load the pretrained tokenizer
        pretrained_weights = args.model
        print(' --------------- token model -->', pretrained_weights)
      
        
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_weights, local_files_only=False) # by cass ood time dataholders.py (, local_files_only=True)
        label_input_indexd_dict = {}

        for id_in_num_format in unique_labels_num_format:
            print('id_in_num_format   ====>', id_in_num_format)
            label_input_id_index = self.tokenizer(str(id_in_num_format))['input_ids']

            print(f"==>> label_input_id_index: {label_input_id_index}")
            label_input_id_index = [x for x in label_input_id_index if x > 1] 
            print(f"==>> label_input_id_index: {label_input_id_index}")
            label_input_indexd_dict[id_in_num_format] = label_input_id_index
     
        print(f"==>> label_input_indexd_dict: {label_input_indexd_dict}")



        if args.query:
            train = [encode_plusplus_t5(dic, self.tokenizer, max_len, label_input_indexd_dict, dic["document"], dic["query"]) for dic in train]
            dev = [encode_plusplus_t5(dic, self.tokenizer, max_len, label_input_indexd_dict, dic["document"], dic["query"]) for dic in dev]
            test = [encode_plusplus_t5(dic, self.tokenizer, max_len, label_input_indexd_dict, dic["document"], dic["query"]) for dic in test]

        else:
            train = [encode_plusplus_t5(dic, self.tokenizer, max_len, label_input_indexd_dict, dic["text"]) for dic in train]
            dev = [encode_plusplus_t5(dic, self.tokenizer, max_len, label_input_indexd_dict, dic["text"]) for dic in dev]
            test= [encode_plusplus_t5(dic, self.tokenizer, max_len, label_input_indexd_dict, dic["text"]) for dic in test]

        shuffle_during_iter = True

        if stage != "train": 

            # ###### sort by length for evaluation and rationale extraction
            train = sorted(train, key = lambda x : x["lengths"], reverse = False)
            dev = sorted(dev, key = lambda x : x["lengths"], reverse = False)
            test = sorted(test, key = lambda x : x["lengths"], reverse = True)


            shuffle_during_iter = False
        
        # if return_as_frames:

        #     self.return_as_frames = {
        #         "train" : pd.DataFrame(train),
        #         "dev" : pd.DataFrame(dev),
        #         "test" : pd.DataFrame(test)
        #     }

        

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
def add_random_word(dataset, fixed_set, fix_size):
    text = []
    for single_text in dataset['text']:
        
        if fixed_set == "fixed0":
            p = r.get_random_word() + " " + r.get_random_word() + " " + r.get_random_word() + " " + r.get_random_word()  + " " + r.get_random_word()#  + " " + r.get_random_word()+ " " + r.get_random_word() # fixed 0 --> 4 random  --> add 3
        elif fixed_set == "fixed1":
            p = single_text + " " + r.get_random_word() + " " + r.get_random_word() + " " + r.get_random_word()  + " " + r.get_random_word()#  + " " + r.get_random_word()+ " " + r.get_random_word() # fixed 3 --> three random
        elif fixed_set == "fixed2":
            p = single_text + " " + r.get_random_word() + " " + r.get_random_word()  + " " + r.get_random_word()#  + " " + r.get_random_word()+ " " + r.get_random_word() 
        elif fixed_set == "fixed3":
            p = single_text + " " + r.get_random_word() + " " + r.get_random_word()#  + " " + r.get_random_word()+ " " + r.get_random_word() 
        elif fixed_set == "fixed4":
            p = single_text + " " + r.get_random_word()# + " " + r.get_random_word() + " " + r.get_random_word() 
        else: pass
        
        if fix_size == 6:
            if fixed_set == "fixed5":
                p = single_text + " " + r.get_random_word() + " " + r.get_random_word() 
            elif fixed_set == "fixed6":
                p = single_text + " " + r.get_random_word() 
            elif fixed_set == "fixed7": 
                pass
            else: # for fixed0 to fixed4
                p = p + " " + r.get_random_word() + " " + r.get_random_word()#
            assert len(p.split()) == 7



        text.append(p)
        if fix_size == 4:
            assert len(p.split()) == 5

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
                fix = 4, sample_size = 50,
                ):
        
        assert type(b_size) == int
        self.batch_size = b_size
        """
        loads data for a classification task from preprocessed .csv =
        files in the dataset/data folder
        and returns three dataholders : train, dev, test
        """
        fixed1 = pd.read_csv(f"./extracted_rationales/sst/data/fixed1/{FA_name}-test.csv")
        fixed2 = pd.read_csv(f"./extracted_rationales/sst/data/fixed2/{FA_name}-test.csv")
        fixed3 = pd.read_csv(f"./extracted_rationales/sst/data/fixed3/{FA_name}-test.csv")
        fixed4 = pd.read_csv(f"./extracted_rationales/sst/data/fixed4/{FA_name}-test.csv")
        fixed5 = pd.read_csv(f"./extracted_rationales/sst/data/fixed5/{FA_name}-test.csv")
        fixed6 = pd.read_csv(f"./extracted_rationales/sst/data/fixed6/{FA_name}-test.csv")
        fixed7 = pd.read_csv(f"./extracted_rationales/sst/data/fixed7/{FA_name}-test.csv")


        if fix == 6:
            fixed7 = fixed7[fixed7["text"].str.contains("[CLS]")==False]
            fixed7 = fixed7[fixed7["text"].str.contains("[SEP]")==False]
            #fixed7 = fixed7.sample(sample_size) #####################################################################################  TEST ###############
            fixed7.to_csv(f"./interpolation/sst/fixed{fix}/{FA_name}-sample{sample_size}-AnalysisSamples.csv")
            ids = fixed7["annotation_id"]
            fixed5 = fixed5.loc[fixed5['annotation_id'].isin(ids)]
            fixed6 = fixed6.loc[fixed6['annotation_id'].isin(ids)]
        
        if fix == 4:
            fixed5 = fixed5[fixed5["text"].str.contains("[CLS]")==False]
            fixed5 = fixed5[fixed5["text"].str.contains("[SEP]")==False]
            #fixed5 = fixed5.sample(sample_size)
            fixed5.to_csv(f"./interpolation/sst/fixed{fix}/{FA_name}-sample{sample_size}-AnalysisSamples.csv")
            ids = fixed5["annotation_id"]

        fixed1 = fixed1.loc[fixed1['annotation_id'].isin(ids)]
        fixed2 = fixed2.loc[fixed2['annotation_id'].isin(ids)]
        fixed3 = fixed3.loc[fixed3['annotation_id'].isin(ids)]
        fixed4 = fixed4.loc[fixed4['annotation_id'].isin(ids)]
        fixed0 = fixed1.copy(deep=True)
        #print(' ================> fixed2 len :', len(fixed2))
        

        fixed0 = add_random_word(fixed0, "fixed0", fix_size=fix).to_dict("records")
        fixed1 = add_random_word(fixed1, "fixed1", fix_size=fix).to_dict("records")
        fixed2 = add_random_word(fixed2, "fixed2", fix_size=fix).to_dict("records")        
        fixed3 = add_random_word(fixed3, "fixed3", fix_size=fix).to_dict("records")
        fixed4 = add_random_word(fixed4, "fixed4", fix_size=fix).to_dict("records")
        if fix == 4: fixed5 = fixed5.to_dict("records")
        elif fix ==6: 
            fixed5 = add_random_word(fixed5, "fixed5", fix_size=fix).to_dict("records")
            fixed6 = add_random_word(fixed6, "fixed6", fix_size=fix).to_dict("records")
            fixed7 = fixed7.to_dict("records")
        else:
            print('need to define fix = 4 or 6')


        if fix == 6: max_len = round(max([len(x["text"].split()) for x in fixed7]))
        else: max_len = round(max([len(x["text"].split()) for x in fixed5]))
        
        self.max_len = max_len

        # load the pretrained tokenizer
        pretrained_weights = args.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights) # by cass ood time dataholders.py (, local_files_only=True)

        self.nu_of_labels = len(np.unique([x["label"] for x in fixed4]))
        #print('self.nu_of_labels  ', self.nu_of_labels)


        if fix == 4 :
            pass
        else:
            fixed7 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed7]
            fixed6 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed6]
        fixed5 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed5]
        fixed4 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed4]
        fixed3 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed3]
        fixed2 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed2]
        fixed1 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed1]
        fixed0 = [encode_plusplus_(dic, self.tokenizer, max_len,  dic["text"]) for dic in fixed0]



        shuffle_during_iter = False
        
        if return_as_frames:
            if fix == 4: self.return_as_frames = { "fixed7" : pd.DataFrame(fixed7), "fixed6" : pd.DataFrame(fixed6),
                                                    "fixed5" : pd.DataFrame(fixed5),
                                                    "fixed4" : pd.DataFrame(fixed4),
                                                    "fixed3" : pd.DataFrame(fixed3),
                                                    "fixed2" : pd.DataFrame(fixed2),
                                                    "fixed1" : pd.DataFrame(fixed1),
                                                    "fixed0" : pd.DataFrame(fixed0),
                                                }
            else:        self.return_as_frames = { #"fixed7" : pd.DataFrame(fixed7), "fixed6" : pd.DataFrame(fixed6),
                                                    "fixed5" : pd.DataFrame(fixed5),
                                                    "fixed4" : pd.DataFrame(fixed4),
                                                    "fixed3" : pd.DataFrame(fixed3),
                                                    "fixed2" : pd.DataFrame(fixed2),
                                                    "fixed1" : pd.DataFrame(fixed1),
                                                    "fixed0" : pd.DataFrame(fixed0),
                                                }

        # prepare data-loaders for training
        if fix == 6:
            self.fixed7_loader = DataLoader(
                fixed6,
                batch_size = self.batch_size,
                shuffle = shuffle_during_iter,
                pin_memory = False,
            )

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


