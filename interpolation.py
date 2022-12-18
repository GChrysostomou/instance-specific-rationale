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

data = BERT_HOLDER_interpolation(
    args["data_dir"], 
    stage = "interpolation",
    b_size = 4,
    #b_size = args["batch_size"], # TO FIX CUDA OUT OF MEMORY, MAY NOT WORK
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)

evaluator.faithfulness_experiments_(data)
print('"********* DONE flip experiments on in-domain"')

del data
del evaluator
gc.collect()
torch.cuda.empty_cache()

dataset = str(user_args["dataset"])



# SET 0 = TOP1, TOP2, TOP3, TOP4 ---> original the top4 ration fixed4
# SET 1 = TOP1, TOP2, TOP3, Rand  --> fixed 3
# SET 2 = TOP1, TOP2, Rand, Rand  --> fixed 2
# SET 3 = TOP1, Rand, Rand, Rand  --> fixed 1
# SET 4 = Rand, Rand, Rand, Rand  --> random 4

fixed_rationale_len = 4

folder = os.path.join(os.getcwd(),
                      "extracted_rationales",
                      dataset,
                      "data",
                      "fixed" + str(fixed_rationale_len),
                       )

S0 = pd.read_csv(folder + '/attention-test.csv')[100:].sample(5)
print(S0)

text_S1 = []
text_S2 = []
text_S3 = []
text_S4 = []



S0_suff = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed.npy', 
                        allow_pickle=True).item()

S1_suff = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S1.npy', 
                        allow_pickle=True).item()

ids = list(S1_suff.keys())

S2_importance_scores = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S2.npy', 
                        allow_pickle=True).item()

S3_importance_scores = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S3.npy', 
                        allow_pickle=True).item()

S4_importance_scores = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S4.npy', 
                        allow_pickle=True).item()



def F_i(M_SO, M_S4, M_Si): # M is the metrics score 
    F_i = abs(M_SO-M_Si)/abs(M_SO-M_S4+0.00001)
    return F_i

S0_sufficiencies = []
S1_sufficiencies = []
S2_sufficiencies = []
S3_sufficiencies = []
S4_sufficiencies = []

for id in ids:
    M_S0 = S0_suff.get(id).get('attention').get('sufficiency')
    M_S1 = S1_suff.get(id).get('attention').get('sufficiency')
    M_S2 = S2_importance_scores.get(id).get('attention').get('sufficiency')
    M_S3 = S3_importance_scores.get(id).get('attention').get('sufficiency')
    M_S4 = S4_importance_scores.get(id).get('attention').get('sufficiency')
    
    S0 = F_i(M_S0, M_S4, M_S0)
    S1 = F_i(M_S0, M_S4, M_S1)
    S2 = F_i(M_S0, M_S4, M_S2)
    S3 = F_i(M_S0, M_S4, M_S3)
    S4 = F_i(M_S0, M_S4, M_S4)

    S0_sufficiencies.append(S0)
    S1_sufficiencies.append(S1)
    S2_sufficiencies.append(S2)
    S3_sufficiencies.append(S3)
    S4_sufficiencies.append(S4)


df = pd.DataFrame(list(zip(ids, S0_sufficiencies, S1_sufficiencies, S2_sufficiencies, S3_sufficiencies, S4_sufficiencies)),
               columns =['id', '0', '1', '2', '3', '4'])

df.to_csv('SST_attention_soft_sufficiency_interpolation.csv')




quit()
'''[      -inf 0.07614467 0.03361953 0.01381731 0.01694824 0.01160708
 0.01682331 0.00476329 0.00979104 0.00561564 0.00487841 0.00499885
 0.00257135 0.00282017 0.00283638 0.00786511 0.06252006 0.03839585
 0.01980747 0.01921392 0.03641286 0.02184515 0.02980015 0.02374282
 0.0199705  0.02761515 0.0133762  0.02211065 0.01256411 0.01168522
 0.01450356 0.36185393       -inf       -inf       -inf       -inf
       -inf       -inf       -inf       -inf       -inf       -inf
       -inf       -inf       -inf       -inf       -inf       -inf]'''




quit()


##########  GET M_SO, M_S1, M_S2, , M_S3, M_S4, for the 50 

############# save to a new diction 


# def presentation_of_SET_0(input_id, important_score): # i from 0 to 4 
#     presentation_at_i = input_id_changed

    

# def get_M_score_of_SETi(presentation):





pwd = os.getcwd()


NOISE_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'NOISE-faithfulness-scores-detailed-std_' + str(user_args["std"]) + '.npy') 
topk_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'topk-faithfulness-scores-detailed.npy')
ATTENTION_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ATTENTION-faithfulness-scores-detailed.npy')
ZEROOUT_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ZEROOUT-faithfulness-scores-detailed.npy')


TOPk_scores = np.load(topk_scores_file, allow_pickle=True).item()
ZEROOUT_scores = np.load(ZEROOUT_scores_file, allow_pickle=True).item()
ATTENTION_scores = np.load(ATTENTION_scores_file, allow_pickle=True).item()
NOISE_scores = np.load(NOISE_scores_file, allow_pickle=True).item()
#NOISE05_scores = np.load(NOISE05_scores_file, allow_pickle=True).item()

data_id_list = TOPk_scores.keys()
fea_list = ['attention', "scaled attention", "gradients", "ig", "gradientshap", "deeplift"]
FA = 'attention'


D_TOP_Suff = []
D_ATTENTION_Suff = []
D_ZEROOUT_Suff = []
D_NOISE_Suff = []


for FA in fea_list:
    Diag_TOP_attention = 0
    Diag_ATTENTION_attention = 0
    Diag_ZEROOUT_attention = 0
    Diag_NOISE_attention = 0
    
    for i, data_id in enumerate(data_id_list):

        top_random_suff_score = TOPk_scores.get(data_id).get('random').get('sufficiency aopc').get('mean')
        NOISE_random_suff_score = NOISE_scores_file.get(data_id).get('random').get('sufficiency')#.get('mean')
        ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get('sufficiency')#.get('mean')
        ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get('sufficiency')#.get('mean')

        top_suff_score = TOPk_scores.get(data_id).get(FA).get('sufficiency aopc').get('mean') # evinf @ 0.1
        if top_suff_score >= top_random_suff_score: Diag_TOP_attention += 1
        else: pass

        NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get('sufficiency')
        if NOISE_suff_score >= NOISE_random_suff_score: Diag_NOISE_attention += 1
        else: pass

        ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get('sufficiency')
        if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: Diag_ZEROOUT_attention += 1
        else: pass

        ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get('sufficiency')
        if ATTENTION_suff_score >= ATTENTION_random_suff_score: Diag_ATTENTION_attention += 1
        else: pass

        
    D_TOP = Diag_TOP_attention/len(data_id_list)
    D_TOP_Suff.append(D_TOP)

    D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
    D_ATTENTION_Suff.append(D_ATTENTION)

    D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
    D_ZEROOUT_Suff.append(D_ZEROOUT)

    D_NOISE1= Diag_NOISE_attention/len(data_id_list)
    D_NOISE_Suff.append(D_NOISE1)


df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ATTENTION_Suff, D_ZEROOUT_Suff, D_NOISE_Suff)),
               columns =['Feature', 'TopK', 'Soft(ATTENTION)', 'Soft(ZEROOUT)', 'Soft(NOISE1)'])

fname = os.path.join(pwd, 'Diagnosticity', str(dataset), 'Diagnosticity_Suff.csv')
os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)
print(df)
df.to_csv(fname)





################ comp

D_TOP_Suff = []
D_ATTENTION_Suff = []
D_ZEROOUT_Suff = []
D_NOISE_Suff = []

for FA in fea_list:
    Diag_TOP_attention = 0
    Diag_ATTENTION_attention = 0
    Diag_ZEROOUT_attention = 0
    Diag_NOISE_attention = 0
    
    for i, data_id in enumerate(data_id_list):

        top_random_suff_score = TOPk_scores.get(data_id).get('random').get('comprehensiveness aopc').get('mean')
        NOISE_random_suff_score = NOISE_scores.get(data_id).get('random').get('comprehensiveness')#.get('mean')
        ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get('comprehensiveness')#.get('mean')
        ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get('comprehensiveness')#.get('mean')

        top_suff_score = TOPk_scores.get(data_id).get(FA).get('comprehensiveness aopc').get('mean') # evinf @ 0.1
        if top_suff_score >= top_random_suff_score: Diag_TOP_attention += 1
        else: pass

        NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get('comprehensiveness')
        if NOISE_suff_score >= NOISE_random_suff_score: Diag_NOISE_attention += 1
        else: pass

        ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get('comprehensiveness')
        if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: Diag_ZEROOUT_attention += 1
        else: pass

        ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get('comprehensiveness')
        if ATTENTION_suff_score >= ATTENTION_random_suff_score: Diag_ATTENTION_attention += 1
        else: pass

    D_TOP = Diag_TOP_attention/len(data_id_list)
    D_TOP_Suff.append(D_TOP)

    D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
    D_ATTENTION_Suff.append(D_ATTENTION)

    D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
    D_ZEROOUT_Suff.append(D_ZEROOUT)

    D_NOISE= Diag_NOISE_attention/len(data_id_list)
    D_NOISE_Suff.append(D_NOISE1)



df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ATTENTION_Suff, D_ZEROOUT_Suff, D_NOISE_Suff, )),
               columns =['Feature', 'TopK', 'Soft(ATTENTION)', 'Soft(ZEROOUT)', 'Soft(NOISE1)'])

fname = os.path.join(pwd, 'Diagnosticity', str(dataset), 'Diagnosticity_Comp.csv')
os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)
df.to_csv(fname)



