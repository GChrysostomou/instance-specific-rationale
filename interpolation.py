#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  CUDA_LAUNCH_BLOCKING=1
import re
from matplotlib.ft2font import FIXED_SIZES
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import argparse
import json
import logging
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


def normal_importance(importance_scores, normalise=5):
    if normalise == 1:
        importance_scores = torch.sigmoid(importance_scores) # 偏大, hover 0.5 SUFF works
    elif normalise == 5:
        importance_scores[torch.isinf(importance_scores)] = -1
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
    else:pass
    return importance_scores




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
    "--FA_name",   
    type = str, 
    help = "directory to save models", 
    default="gradients" 
    #[random 'attention', "scaled attention", "gradients", "ig", "deeplift"]
)


parser.add_argument(
    "--sample_size",   
    type = int, 
    help = "directory to save extracted_rationales", 
    default = 500,
)

parser.add_argument(
    "--fix_size",   
    type = int, 
    help = "directory to save extracted_rationales", 
    default = 6,
)

user_args = vars(parser.parse_args())
log_dir = "experiment_logs/evaluate_" + user_args["dataset"] + "_" +  date_time + "/"
config_dir = "experiment_config/evaluate_" + user_args["dataset"] + "_" +  date_time + "/"
os.makedirs(log_dir, exist_ok = True)
os.makedirs(config_dir, exist_ok = True)
import config.cfg
config.cfg.config_directory = config_dir
logging.basicConfig(filename= log_dir + "/out.log", format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()
from src.common_code.initialiser import initial_preparations
# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "evaluate")
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




figsize1, figsize2 = 4, 3

FA_name = args["FA_name"]
sample_size =  args["sample_size"]
fix_size = args["fix_size"]

total_len = fix_size 

def F_i(M_SO, M_S4, M_Si): # M is the metrics score 
    F_i = abs(M_SO-M_Si)/abs(M_SO-M_S4+0.0001)
    return F_i


data = BERT_HOLDER_interpolation(args["data_dir"], stage = "interpolation", b_size = 8, 
                                        FA_name = FA_name, fix = fix_size, sample_size = sample_size)

if fix_size == 4:
    loader_list = [ data.fixed0_loader,
                    data.fixed1_loader,
                    data.fixed2_loader,
                    data.fixed3_loader,
                    data.fixed4_loader,
                    # data.fixed5_loader,
                    # data.fixed6_loader,
                    # data.fixed7_loader,
                    ]
else:
    loader_list = [data.fixed0_loader,
                    data.fixed1_loader,
                    data.fixed2_loader,
                    data.fixed3_loader,
                    data.fixed4_loader,
                    data.fixed5_loader,
                    data.fixed6_loader,
                    #data.fixed7_loader,
                    ]
    

comp_list = []
comp_list2 = []

for dataloader_i, data_loader in enumerate(loader_list):


    fname2 = os.path.join(
            os.getcwd(),
            args["model_dir"],
        )
    fname2 = glob.glob(fname2 + "*output*25.npy")[0]
    original_prediction_output = np.load(fname2, allow_pickle = True).item()

    comp_total = torch.tensor([])
    comp_total2 = torch.tensor([])


    for i, batch in enumerate(data_loader):




        ############ get the importance scores and pad random ones ##########
        model.eval()

        batch = {"annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "query_mask" : batch["query_mask"].squeeze(1).to(device),
                "special_tokens" : batch["special tokens"],
                "retain_gradient" : False,
                "importance_scores": batch["importance_scores"],
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


        ## baseline sufficiency # input all zero
        #batch["input_ids"] = torch.zeros_like(batch["input_ids"]).long() 
        yhat, _  = model(**batch)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]
        #suff_y_zero = sufficiency_(full_text_probs, reduced_probs)
        comp = comprehensiveness_(full_text_probs, reduced_probs)

        comp_total = np.concatenate((comp_total, comp),axis=0)
        
     


######################### OURS ###################

        if dataloader_i == 0:
            paded_IS = torch.rand(batch["input_ids"].squeeze(1).size()) # all 
        else:
            for n, one_list in enumerate(batch["importance_scores"]):
                one_list = one_list[1:] # remove "["" 
                one_list = one_list[:-1]
                floats = [float(x) for x in one_list.split()]
                if n == 0: 
                    IS = torch.tensor(floats).unsqueeze(0)
                else:
                    one_list = torch.tensor(floats).unsqueeze(0)
                    IS = torch.cat((IS, one_list), 0) 
            # pad zero for random words
            to_pad_num = fix_size+1-IS.size()[1] ######### need to change if testing !!!!!!!!
            pad = torch.zeros(IS.size()[0], to_pad_num) # len(loader_list)=6
            paded_IS = torch.cat((IS,pad), dim = 1)


        
        batch["add_noise"] = True
        batch["input_ids"] = original_sentences.to(device)
        batch["faithful_method"] = "soft_suff"
        
        if batch["faithful_method"] == "soft_suff":
            normal = 1
        else: normal = 5 # 1 for Suff and 5 for Comp
        batch["importance_scores"]= normal_importance(paded_IS, normal).to(device)

        batch["rationale_mask"] = torch.ones(batch["input_ids"].shape).to(device), # all out
        model2.eval()
        ##### 进 model 前, rationale 已经因为comp 被删掉了
        
        yhat, _  = model2(**batch)
        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()
        reduced_probs = yhat[rows, full_text_class]
        comp2 = comprehensiveness_(full_text_probs, reduced_probs)
        
        comp_total2 = np.concatenate((comp_total2, comp2),axis=0)


    comp_final2 = np.mean(comp_total2)
    comp_final = np.mean(comp_total)
    comp_list.append(comp_final)  
    comp_list2.append(comp_final2)





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


if fix_size == 4:
    set = ['0', '1', '2', '3', '4']#, '5', '6'
elif fix_size == 6:
    set = ['0', '1', '2', '3', '4', '5', '6']#, '5', '6'
else:
    print(' ---- ')
df = pd.DataFrame(list(zip(set, F_comp, F_comp2, comp_list, comp_list2)), 
                        columns = ['Set', 'F-Comp', 'F-SoftComp', 'Comprehensiveness', 'Soft-Comprehensiveness'])

df.to_csv(f'./interpolation/{args["dataset"]}/fixed6/{args["dataset"]}_{FA_name}_full.csv')





comp = df["F-Comp"]
soft = df["F-SoftComp"]
SET=df.index
# Initialize figure and axis

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(figsize1, figsize2)) # gradients 3

# Plot lines
ax.plot(SET, comp, color="gray", label='Comp')
ax.plot(SET, soft, color="red", label='Soft-Comp')

if FA_name=='random':
    ax.set_xlabel('Replaced tokens')
    ax.set_ylabel('f(i) = |M(So)-M(Si)| / |M(So)-M(S6)|')
    ax.set_title(str(FA_name).capitalize(), fontsize=18)
else:
    # ax.set_xlabel('Replaced tokens')
    # ax.set_ylabel('f(i) = |M(So)-M(Si)| / |M(So)-M(S6)|')
    ax.set_title(str(FA_name).capitalize(), fontsize=20)

ax.legend()
import matplotlib
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(figsize1+0.5, figsize2+0.5)
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.25)
# plt.gcf().subplots_adjust(right=-0.05)
plt.show()
fig.savefig(f'./interpolation/sst/fixed{fix_size}/{FA_name}_fullsample_plot.png')