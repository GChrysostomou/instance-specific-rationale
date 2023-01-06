#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import argparse
import logging
import gc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import datetime
import os


date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst", 
    #choices = ["SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr"]
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
    default = "fixed",
    choices = ["contigious", "topk", "fixed"]
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = None, 
    choices = [None]
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

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "extract")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")

from src.data_functions.dataholder import BERT_HOLDER as dataholder
from src.evaluation import evaluation_pipeline

data = dataholder(
    args["data_dir"], 
    b_size = args["batch_size"],
    stage = "eval",
    return_as_frames = True
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)


logging.info("*********extracting in-domain rationales")

evaluator.register_importance_(data, data_split_name="test", tokenizer=None, max_seq_len=None)
#evaluator.create_rationales_(data)
evaluator.create_rationales_interpolation(data, fixed_rationale_len=7)
evaluator.create_rationales_interpolation(data, fixed_rationale_len=6)
evaluator.create_rationales_interpolation(data, fixed_rationale_len=5)
evaluator.create_rationales_interpolation(data, fixed_rationale_len=4)  # generate data for interpolation 
evaluator.create_rationales_interpolation(data, fixed_rationale_len=3)  # generate data for interpolation 
evaluator.create_rationales_interpolation(data, fixed_rationale_len=1)  # generate data for interpolation 
evaluator.create_rationales_interpolation(data, fixed_rationale_len=2)  # generate data for interpolation 


del data
del evaluator
gc.collect()#


