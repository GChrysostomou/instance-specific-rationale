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
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.common_code.metrics import comprehensiveness_, normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_, normalized_comprehensiveness_soft_, normalized_sufficiency_soft_
from sklearn.metrics import classification_report

torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


from src.evaluation.experiments.rationale_extractor import rationale_creator_, rationale_creator_interpolation_, extract_importance_, extract_shap_values_
from src.data_functions.dataholder import BERT_HOLDER_interpolation
from src.evaluation import evaluation_pipeline
from src.models.bert import BertClassifier_zeroout, bert, BertClassifier_attention
from src.common_code.useful_functions import batch_from_dict_, create_only_query_mask_, create_rationale_mask_ # batch_from_dict --> batch_from_dict_

model = bert(output_dim = 2)
model.load_state_dict(torch.load("./trained_models/sst/bert25.pt", map_location=device))
model.to(device)

model2 = BertClassifier_zeroout(output_dim = 2)
model2.load_state_dict(torch.load("./trained_models/sst/bert25.pt", map_location=device))
model2.to(device)


def F_i(M_SO, M_S4, M_Si): # M is the metrics score 
    F_i = abs(M_SO-M_Si)/abs(M_SO-M_S4+0.00001)
    return F_i


## testing different FA!!!!!!!!
FA_name = "scaled attention" #['attention', "scaled attention", "gradients", "ig", "deeplift"]


data = BERT_HOLDER_interpolation(args["data_dir"], stage = "interpolation",b_size = 4, FA_name = FA_name)

loader_list = [
            data.fixed0_loader,
            data.fixed1_loader,
            data.fixed2_loader,
            data.fixed3_loader,
            data.fixed4_loader,
            data.fixed5_loader,
            data.fixed6_loader,
            ]
    

comp_list = []
comp_list2 = []
for i, data_loader in enumerate(loader_list):


    ########## register importance scores 
    # #evaluator.register_importance_(data, data_split_name="test")
    # extract_importance_(
    #                 model = model, 
    #                 data_split_name = "test",
    #                 data = data,
    #                 model_random_seed = 25,
    #             )

    # extract_shap_values_(
    #             model = model, 
    #             data = data,
    #             data_split_name = "test",
    #             model_random_seed = 25,
    #             # no_of_labels = no_of_labels,
    #             # max_seq_len = max_seq_len,
    #             # tokenizer = tokenizer
    #         )



    print(' ++++++++++++++++++ data_loader', i )
    print('+++++++++++++++++++++++++++++++++++')
    fname2 = os.path.join(
            os.getcwd(),
            args["model_dir"],
        )
    fname2 = glob.glob(fname2 + "*output*25.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    comp_total = torch.tensor([])
    comp_total2 = torch.tensor([])



    for i, batch in enumerate(data_loader):

        IS = torch.zeros(batch["input_ids"].squeeze(1).size())

        for i, one_list in enumerate(batch["importance_scores"]):
                
            one_list = one_list[1:]
            one_list = one_list[:-1]
            floats = [float(x) for x in one_list.split()]

            if i == 0: # 都不重要
                IS = torch.tensor(floats).unsqueeze(0)
            else:
                one_list = torch.tensor(floats).unsqueeze(0)
                IS = torch.cat((IS, one_list), 0) 
    
        # print("==>> type(IS): ", (IS))
        # print(" =========    input_ids     =====", batch["input_ids"])
        pad = torch.zeros(IS.size()[0], len(loader_list)-1-IS.size()[1])
        paded_IS = torch.cat((IS,pad), dim = 1)
         


        importance_scores = F.pad(input=IS, pad=batch["input_ids"].squeeze(1).size(), mode='constant', value=0)
        print("==>> 2  importance_scores: ", importance_scores)
        
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
                "importance_scores": paded_IS.to(device),
                }
        print( "==>>  3 batch[importance_scores]: ", batch["importance_scores"].shape )
        
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

        batch["input_ids"] = torch.zeros_like(batch["input_ids"]).long()

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
                        rationale_mask = torch.ones(batch["input_ids"].shape).to(device), 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                    )
        comp_total = np.concatenate((comp_total, comp),axis=0)
        print(' HAPPY £££££££  comp_total', comp_total)



######################### OURS ###################
        batch["faithful_method"] = rationale_mask
        batch["faithful_method"] = "soft_comp"
        batch["add_noise"] = True
        batch["rationale_mask"] = torch.ones(batch["input_ids"].shape).to(device), 


        yhat, _  = model2(**batch)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs,
        )
        print('  ')
        print('  ')
        print( "==>>  4 batch[importance_scores]: ", batch["importance_scores"].shape )
        print(  batch["importance_scores"])
        #comp2, comp_probs2  = normalized_comprehensiveness_soft_(
        comp2, comp_probs2  = normalized_comprehensiveness_soft_(
                        model = model2.to(device), 
                        original_sentences = original_sentences.to(device), 
                        #rationale_mask = torch.ones(batch["input_ids"].shape).to(device), 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        importance_scores = batch["importance_scores"],
                        rows = rows,
                        suff_y_zero = suff_y_zero,
                        use_topk=True,
                        normalise =4,
                        #only_query_mask = None,

                    )
        comp_total2 = np.concatenate((comp_total2, comp2),axis=0)
        #print(' comp_total2 ', comp_total2)

    #quit()            
    comp_final = np.mean(comp_total)
    comp_list.append(comp_final)
    comp_final2 = np.mean(comp_total2)
    comp_list2.append(comp_final2)

    print(' comp list  ', comp_list, comp_list2)




M_SO = comp_list[0]
M_S6 = comp_list[-1]
F_comp = []
for comp in comp_list:
    Fi = F_i(M_SO, M_S6, comp)
    F_comp.append(Fi)

M_SO = comp_list2[0]
M_S6 = comp_list2[-1]
F_comp2 = []
for comp in comp_list2:
    Fi = F_i(M_SO, M_S6, comp)
    F_comp2.append(Fi)

set = ['0', '1', '2', '3', '4', '5', '6']
df = pd.DataFrame(list(zip(set, F_comp, F_comp2, comp_list, comp_list2)), 
                        columns = ['Set', 'F-Comp', 'F-SoftComp', 'Comprehensiveness', 'Soft-Comprehensiveness'])

df.to_csv(f'./interpolation/{args["dataset"]}/fixed6/{args["dataset"]}_{FA_name}.csv')





comp = df["F-Comp"]
soft = df["F-SoftComp"]
SET=df.index
# Initialize figure and axis
fig, ax = plt.subplots(figsize=(5, 5))

# Plot lines
ax.plot(SET, comp, color="red")
ax.plot(SET, soft, color="green")

# Fill area when income > expenses with green
ax.fill_between(
    SET, comp, soft, where=(soft >= comp), 
    interpolate=True, color="green", alpha=0.25, 
    label="Soft Comprehensiveness"
)

# Fill area when income <= expenses with red
ax.fill_between(
    SET, comp, soft, where=(soft < comp), 
    interpolate=True, color="red", alpha=0.25,
    label="Comprehensiveness"
)

ax.set_xlabel('Replaced tokens')
ax.set_ylabel('f(i) = |M(So)-M(Si)| / |M(So)-M(S6)|')
ax.set_title(f'Interpolation Analysis ({FA_name})')

ax.legend()
plt.show()
fig.savefig(f'./interpolation/sst/fixed6/{FA_name}_plot.png')