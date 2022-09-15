#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import pandas as pd
import argparse
import json
import logging
import gc
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import datetime


date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "evinf", 
    #choices = ["sst", "evinf", "multirc", "agnews"]
)

parser.add_argument(
    "--data_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)

parser.add_argument(
    "--model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "trained_models/"
)

parser.add_argument(
    "--extracted_rationale_dir",   
    type = str, 
    help = "directory to save extracted_rationales", 
    default = "extracted_rationales/"
)

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    "--rationale_length", 
    type = str, 
    help = "set to instance-specific if you want to calculate instance level rationale length", 
    default = "fixed",
    choices = ["fixed", "instance-specific"]
)

parser.add_argument(
    "--divergence", 
    type = str, 
    help = "divergence metric used to compute variable rationales", 
    default = "jsd",
    choices = ["jsd", "kldiv", "perplexity", "classdiff"]
)

parser.add_argument(
    '--extract_double', 
    help='for testing at larger rationale lengths', 
    action='store_true'
)

user_args = vars(parser.parse_args())

log_dir = "experiment_logs/extract_" + user_args["dataset"] + "_" +  date_time + "/"
config_dir = "experiment_config/extract_" + user_args["dataset"] + "_" + date_time + "/"


os.makedirs(log_dir, exist_ok = True)
os.makedirs(config_dir, exist_ok = True)

import config.cfg

config.cfg.config_directory = config_dir

logging.basicConfig(
    filename= log_dir + "/out.log", 
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.common_code.initialiser import initial_preparations
import datetime
import sys

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "extract")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")

from src.data_functions.dataholder import classification_dataholder 
from src.evaluation import evaluation_pipeline

data = classification_dataholder(
    args["data_dir"], 
    b_size = args["batch_size"],
    return_as_frames = True,
    stage = "extract",
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)


print(' ============================ ')
print(' prepare_for_rationale_creation_ ')
evaluator.prepare_for_rationale_creation_(data)

evaluator.register_importance_(data, data_split_name='test', no_of_labels=data.nu_of_labels, max_seq_len=data.max_len, tokenizer=data.tokenizer)
evaluator.create_rationales_(data)

del data
del evaluator
gc.collect()
torch.cuda.empty_cache()



###################


dataset = user_args["dataset"]
data_split_name = 'dev'
best_model = glob.glob(f'trained_models/{dataset}/*.pt')
print('best model: ', best_model)
best_model_num = str(best_model)[-7:-5]
print(best_model_num)

# likelihood_meta_result_path = glob.glob(f'./trained_models/{dataset}/*seed*{best_model_num}.npy')[0]
# likelihood_meta = np.load(likelihood_meta_result_path, allow_pickle= True)

'''
{'test_82': {'predicted': array([-0.72803926, -1.0137744 ], dtype=float32), 'actual': 1}, 
'test_156': {'predicted': array([-3.1670673,  3.6749783], dtype=float32), 'actual': 1}, 
'test_1701': {'predicted': array([ 1.3215427, -2.890399 ], dtype=float32), 'actual': 0}, 
'test_1211': {'predicted': array([ 1.4527382, -3.0300527], dtype=float32), 'actual': 0}}
'''

# meta = f'extracted_rationales/{dataset}/topk/dev-rationale_metadata.npy'
# meta = np.load(meta,allow_pickle=True)
fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )
#fname += f"{data_split_name}_importance_scores_{best_model_num}.npy"

importance_score_path = glob.glob(fname+f'{data_split_name}*{best_model_num}*')[0]

## retrieve importance scores
importance_scores = np.load(importance_score_path, allow_pickle = True).item() # dictionary


print('===========================================')
#print(importance_scores)
print(importance_scores.keys())  # 'dev_624', 'dev_672', 'dev_740', 'dev_801', 'dev_845'
                                # '3897026_1', '3897026_4', '3897026_5'
print(importance_scores.get('dev_624').keys()) #dict_keys(['random', 'attention', 'gradients', 'ig', 'scaled attention', 'lime', 'deeplift'])
print('===========================================')
print(len(importance_scores.get('dev_624').get('deeplift')))
# print(len(importance_scores.get('3897026_4').get('deeplift')))
# print(len(importance_scores.get('3897026_5').get('deeplift')))
print(len(importance_scores.get('dev_624').get('random')))
print(importance_scores.get('dev_624').get('deeplift')) # [          -inf  1.52082415e-03  2.31635128e-03  .....] 长度等于上面的长度
print('===========================================')
#print(importance_scores.get('dev_740'))
# print('===========================================')
# print(meta.keys())


# ask georgy, if removed it / stopwords from rationales and then take out rationales
# the length of importance score (non -inf) is not equal to the text length

'''
>> sorted_importance_score = list(numpy.argsort(attention_importance_scores)).reverse()
>>> sorted_importance_score
>>> sorted_importance_score = numpy.argsort(attention_importance_scores)

'''

import pprint




data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        stage = "train"
    )

test_stats = test_predictive_performance_rank(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        save_output_probs = True
    )   

# logging.info( 
#     pprint.pformat(data_desc, indent = 4)
# )