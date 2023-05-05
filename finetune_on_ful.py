#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os 
import argparse
import logging
import datetime
import gc

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "spanish_xnli", 
    # choices = ["french_xnli" "french_paws" french_csl # spanish_csl
    #choices = ["ant", "csl","ChnSentiCorp", "sst", "evinf", "agnews", "multirc", "evinf_FA"]
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
    help = "directory to save models, mannually modify it for multi and mono", 
    default = "test_trained_models/"  # macbert bert zhbert french_bert
)


parser.add_argument(
    "--seed",   
    type = int, 
    help = "random seed for experiment",
    default = 15
)


parser.add_argument(
    '--evaluate_models', 
    help='test predictive performance in and out of domain', 
    action='store_true',
    default=False,
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None





### used only for data stats
data_dir_plain = user_args["data_dir"]


log_dir = "experiment_logs/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" +  date_time + "/"
config_dir = "experiment_config/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" + date_time + "/"


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


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

logging.info("Running on cuda : {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from src.common_code.initialiser import initial_preparations
import datetime

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "train")



logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")



if args['model_abbreviation'] == 't5m': 
    from src.data_functions.dataholder import mT5_HOLDER as dataholder
    print(' ')
    print(' ')
    print('using T5')
else: from src.data_functions.dataholder import BERT_HOLDER as dataholder
from src.tRpipeline import train_and_save, train_and_save_t5, test_predictive_performance, keep_best_model_
from src.data_functions.useful_functions import describe_data_stats


data_desc = describe_data_stats(
    path_to_data = args["data_dir"],
    path_to_stats = os.path.join(
        data_dir_plain,
        args["dataset"],
        ""
    ) 
)

import pprint

logging.info( 
    pprint.pformat(data_desc, indent = 4)
)

del data_desc
gc.collect()

# training the models and evaluating their predictive performance
# on the full text length

data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        stage = "train"
    )


print(' done loading data')

## evaluating finetuned models
if args["evaluate_models"]:

    ## in domain evaluation
    test_stats = test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
    )    

    del data
    gc.collect()

    ## shows which model performed best on dev F1 (in-domain)
    ## if keep_models = False then will remove the rest of the models to save space
    keep_best_model_(keep_models = False)

else:

    if args['model_abbreviation'] == 't5m': 

        train_and_save_t5(
        train_data_loader = data.train_loader, 
        dev_data_loader = data.dev_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        ) 


    
    else: train_and_save(
        train_data_loader = data.train_loader, 
        dev_data_loader = data.dev_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
    ) 