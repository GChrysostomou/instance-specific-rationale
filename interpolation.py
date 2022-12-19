#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
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
import glob

from src.common_code.metrics import comprehensiveness_, normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_, normalized_comprehensiveness_soft_, normalized_sufficiency_soft_
from sklearn.metrics import classification_report

torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(' ---------> ', device)
CUDA_LAUNCH_BLOCKING=1



date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst",
    # choices = ["agnews","evinf", "sst","multirc",]
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
    default="trained_models/"
)


parser.add_argument(
    "--evaluation_dir",   
    type = str, 
    help = "directory to save faithfulness results", 
    default = "posthoc_results/"
)

parser.add_argument(
    "--extracted_rationale_dir",   
    type = str, 
    help = "directory to save extracted_rationales", 
    default = "extracted_rationales/"
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = None, 
    choices = [None, "kuma", "rl"]
)

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
CUDA_LAUNCH_BLOCKING=1

logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()
CUDA_LAUNCH_BLOCKING=1

from src.common_code.initialiser import initial_preparations

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "evaluate")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")



from src.data_functions.dataholder import BERT_HOLDER_interpolation
from src.evaluation import evaluation_pipeline
from src.models.bert import BertClassifier_noise, BertClassifier_zeroout, bert, BertClassifier_attention
from src.common_code.useful_functions import batch_from_dict_, create_only_query_mask_, create_rationale_mask_ # batch_from_dict --> batch_from_dict_

FA_name = "attention"
data = BERT_HOLDER_interpolation(
    args["data_dir"], 
    stage = "interpolation",
    b_size = 4,
    FA_name = FA_name,
)

model = bert(output_dim = 2)
model.load_state_dict(torch.load("./trained_models/sst/bert25.pt", map_location=device))
model.to(device)

model2 = BertClassifier_noise(output_dim = 2)
model2.load_state_dict(torch.load("./trained_models/sst/bert25.pt", map_location=device))
model2.to(device)


comp_list = []
comp_list2 = []
for data in [data.fixed4_loader,data.fixed3_loader,data.fixed2_loader,data.fixed1_loader,data.fixed0_loader]:
    
    fname2 = os.path.join(
            os.getcwd(),
            args["model_dir"],
        )
    fname2 = glob.glob(fname2 + "*output*25.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    comp_total = torch.tensor([])
    for i, batch in enumerate(data):
        # print( '         ==========')

        # print(batch["importance_scores"])
        # print(batch["importance_scores"].dtype())

        IS = torch.tensor(batch["input_ids"].squeeze(1).size())
        for i, one_list in enumerate(batch["importance_scores"]):
            print(one_list)
            one_list = one_list[1:]
            one_list = one_list[:-1]
            print(one_list)

            floats = [float(x) for x in one_list.split()]
            one_list = torch.tensor(floats)
            IS[i,:] = one_list

            
        model.eval()
        model.zero_grad()
        batch = {"annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "query_mask" : batch["query_mask"].squeeze(1).to(device),
                "special_tokens" : batch["special tokens"],
                "retain_gradient" : False,
                "importance_scores": IS.to(device),
                }
        
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
    
        original_prediction =  batch_from_dict_(
                        batch_data = batch, 
                        metadata = original_prediction_output, 
                        target_key = "predicted",
                    )  # return torch.tensor(new_tensor).to(device)


        ## prepping for our experiments
        original_sentences = batch["input_ids"].clone().detach()
        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))

        only_query_mask=torch.zeros_like(batch["input_ids"]).long()
        batch["input_ids"] = only_query_mask

        
        yhat, _  = model(**batch)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        rationale_mask = torch.zeros(original_sentences.size())
        comp, comp_probs  = normalized_comprehensiveness_(
                        model = model, 
                        original_sentences = original_sentences.to(device), 
                        rationale_mask = rationale_mask.to(device), 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        #suff_y_zero = suff_y_zero,
                        comp_y_one=1-suff_y_zero,
                    )
        comp_total = np.concatenate((comp_total, comp),axis=0)
        
        # print(' -----------  ')
        # print(list(batch["importance_scores"]))

        # importance_scores_for_soft = []
        # for i in list(batch["importance_scores"]):
            
        #     # print(' ')
        #     # print(' ')
        #     # print(' --------------  ')
        #     # print(i)
        #     i = re.sub("[^0-9.]", "", i)
        #     importance_scores_for_soft.append(float(i)) 
        
        comp2, comp_probs2  = normalized_comprehensiveness_soft_(
                        model = model2, 
                        original_sentences = original_sentences.to(device), 
                        rationale_mask = rationale_mask.to(device), 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        importance_scores = torch.FloatTensor(batch["importance_scores"]).to(device),
                        rows = rows,
                        #suff_y_zero = suff_y_zero,
                        comp_y_one=1-suff_y_zero,
                        use_topk=True,
                    )
        comp_total2 = np.concatenate((comp_total2, comp2),axis=0)

                
    comp_final = np.mean(comp_total)
    comp_list.append(comp_final)
    comp_final2 = np.mean(comp_total2)
    comp_list2.append(comp_final2)

print(comp_list)
print(comp_list2)
set = ['S0','S1','S2','S3','S4']  # removed 4 ----> remove 0
df = pd.DataFrame(list(zip(set, comp_list, comp_list2)), coumns = ['Set', 'Comprehensiveness', 'Soft-Comprehensiveness'])
df.to_csv('interpolation_on_sst_attention.csv')
quit()




# SET 0 = TOP1, TOP2, TOP3, TOP4 ---> original the top4 ration fixed4
# SET 1 = TOP1, TOP2, TOP3, Rand  --> fixed 3
# SET 2 = TOP1, TOP2, Rand, Rand  --> fixed 2
# SET 3 = TOP1, Rand, Rand, Rand  --> fixed 1
# SET 4 = Rand, Rand, Rand, Rand  --> random 4




def F_i(M_SO, M_S4, M_Si): # M is the metrics score 
    F_i = abs(M_SO-M_Si)/abs(M_SO-M_S4+0.00001)
    return F_i



