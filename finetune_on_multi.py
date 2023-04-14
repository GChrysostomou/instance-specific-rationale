#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os 
import argparse
import logging
import datetime
import gc
from src.common_code.initialiser import initial_preparations
import config.cfg

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])
abb_dict = {"bert-base-uncased": "bert", 
            "allenai/scibert_scivocab_uncased": "scibert",
            "roberta-base": "roberta",
            "facebook/m2m100_418M": "m2m",
            "xlm-roberta-base": "xlm_roberta",
            }

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "evinf", 
    #choices = ["sst", "evinf", "agnews", "multirc", "evinf_FA"]
)

parser.add_argument(
    "--if_multi", 
    type = str, 
    help = "if using multilingual model", 
    default = True, 
    choices = [True, False]
)

parser.add_argument(
    "--multi_model_name", 
    type = str, 
    help = "if using multilingual model", 
    default = 'xlm-roberta-base',
    choices = ['xlm-mlm-100-1280', 'xlm-roberta-base', 'facebook/m2m100_418M'],
)

parser.add_argument(
    "--model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "multilingual_trained_models/"
)

parser.add_argument(
    "--data_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)


parser.add_argument(
    "--seed",   
    type = int, 
    help = "random seed for experiment",
    default = 10
)

parser.add_argument(
    '--evaluate_models', 
    help='test predictive performance in and out of domain', 
    action='store_true',
    default=False,
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None


if user_args['multi_model_name'] != None:
    user_args.update({"model":str(user_args['multi_model_name'])})
    user_args.update({"multi_model":str(user_args['multi_model_name'])})
    user_args.update({"model_abbreviation":str(abb_dict[user_args['multi_model_name']])})
if user_args['if_multi'] != True:
    user_args.update({"model_dir":str('multilingual_'+user_args['model_dir'])})
print(user_args)

### used only for data stats
data_dir_plain = user_args["data_dir"]


log_dir = "experiment_logs/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" +  date_time + "/"
config_dir = "experiment_config/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" + date_time + "/"



os.makedirs(log_dir, exist_ok = True)
os.makedirs(config_dir, exist_ok = True)



config.cfg.config_directory = config_dir
args = initial_preparations(user_args, stage = "train")


logging.basicConfig(
                    filename= log_dir + "/out.log", 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S'
                  )




# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():device = torch.device("cuda:0")
else:device = torch.device("cpu")
print("running on the ", device)
logging.info("Running on cuda : {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# creating unique config from stage_config.json file and model_config.json file




logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")



from src.data_functions.dataholder import BERT_HOLDER 
from src.data_functions.dataholder import multi_BERT_HOLDER #as multi_dataholder
from src.tRpipeline import multi_train_and_save, multi_test_predictive_performance, keep_best_model_
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
import transformers
transformers.logging.set_verbosity_error()
# training the models and evaluating their predictive performance
# on the full text length
if args['multi_model_name'] == 'facebook/m2m100_418M': 
    from transformers import M2M100Tokenizer
    tokenizer = M2M100Tokenizer
    data = multi_BERT_HOLDER(
        self_define_tokenizer = tokenizer,
        path = args["data_dir"], 
        b_size = args["batch_size"],
        stage = "train"
    )
elif args['multi_model_name'] == 'xlm-roberta-base': #[, , 'xlm-mlm-100-1280'],
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer
    data = multi_BERT_HOLDER(
        self_define_tokenizer = tokenizer,
        path = args["data_dir"], 
        b_size = args["batch_size"],
        stage = "train"
    )
else:
    data = BERT_HOLDER(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        stage = "train"
    )


  # return a dictionary, of train/dev/test dataloader

print(' ')
print(' ')
print(' ')
print(' Have loaded dataloader')


## evaluating finetuned models
if args["evaluate_models"]:
    ## in domain evaluation
    test_stats = multi_test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        model_name=args['multi_model_name'],
    )    

    del data
    gc.collect()
    keep_best_model_(keep_models = False)

else:
    print(' ======== ')
    if args['multi_model_name'] == 'xlm-roberta-base':
        from transformers import XLMRobertaConfig, XLMRobertaModel
        multi_train_and_save(
            self_define_model = XLMRobertaModel,
            self_define_config = XLMRobertaConfig,
            train_data_loader = data.train_loader, 
            dev_data_loader = data.dev_loader, 
            for_rationale = False, 
            output_dims = data.nu_of_labels,
            model_name=args['multi_model_name'],
        ) 