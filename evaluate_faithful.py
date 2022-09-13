#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
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
import sys


date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst", 
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
    default = "full_text_models/"
)


parser.add_argument(
    "--evaluation_dir",   
    type = str, 
    help = "directory to save decision flips", 
    default = "faithfulness_metrics/"
)

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    "--extracted_rationale_dir",   
    type = str, 
    help = "directory to save extracted_rationales", 
    default = "extracted_rationales/"
)

parser.add_argument(
    '--extract_double', 
    help='for testing at larger rationale lengths', 
    action='store_true'
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

log_dir = "experiment_logs/evaluate_FAClassifier_faithful" + user_args["dataset"] + "_" +  date_time + "/"
config_dir = "experiment_config/evaluate_FAClassifier_faithful" + user_args["dataset"] + "_" +  date_time + "/"


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
args = initial_preparations(user_args, stage = "evaluate")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")


from src.data_functions.dataholder import classification_dataholder
from src.evaluation import evaluation_pipeline


dataset = user_args["dataset"]
best_model = glob.glob(f'trained_models/{dataset}/*.pt')
print('best model: ', best_model)
best_model_num = str(best_model)[-7:-5]
print(best_model_num)

likelihood_meta_result_path = glob.glob(f'./trained_models/{dataset}/*seed*{best_model_num}.npy')[0]
likelihood_meta = np.load(likelihood_meta_result_path, allow_pickle= True)

'''
{'test_82': {'predicted': array([-0.72803926, -1.0137744 ], dtype=float32), 'actual': 1}, 
'test_156': {'predicted': array([-3.1670673,  3.6749783], dtype=float32), 'actual': 1}, 
'test_1701': {'predicted': array([ 1.3215427, -2.890399 ], dtype=float32), 'actual': 0}, 
'test_1211': {'predicted': array([ 1.4527382, -3.0300527], dtype=float32), 'actual': 0}}
'''

meta = f'extracted_rationales/{dataset}/topk/dev-rationale_metadata.npy'
meta = np.load(meta,allow_pickle=True)
print('===========================================')
print(meta)
print(meta.shape)
print(meta.shape())
print('===========================================')
print(meta.get('dev_740').keys())
print('===========================================')
print(meta.get('dev_740'))
# print('===========================================')
# print(meta.keys())