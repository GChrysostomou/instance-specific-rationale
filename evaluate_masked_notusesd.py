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
import datetime
import sys




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




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
    default = "trained_models/"
)


parser.add_argument(
    "--evaluation_dir",   
    type = str, 
    help = "directory to save decision flips", 
    default = "posthoc_results/"
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

# parser.add_argument(
#     '--extract_double', 
#     help='for testing at larger rationale lengths', 
#     action='store_true'
# )

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

log_dir = "experiment_logs/evaluate_" + user_args["dataset"] + "_" +  date_time + "/"
config_dir = "experiment_config/evaluate_" + user_args["dataset"] + "_" +  date_time + "/"


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
#CUDA_LAUNCH_BLOCKING=1
logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

from src.common_code.initialiser import initial_preparations


# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "evaluate")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")


from src.data_functions.dataholder import BERT_HOLDER
from src.evaluation import evaluation_pipeline

data = BERT_HOLDER(
    args["data_dir"], 
    b_size = 4, # args["batch_size"] 4 
    stage = "evaluate", # stage = "eval",
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)
evaluator.faithfulness_experiments_(data)

'''
evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)

evaluator.faithfulness_experiments_(data)
'''

logging.info("*********conducting in-domain flip experiments")


print(' DONE faithfulness metrics')

evaluator.feature_scoring_performance_()  ## debug by cass

del data
del evaluator
gc.collect()
torch.cuda.empty_cache()

  