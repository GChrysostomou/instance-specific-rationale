
    
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import pandas as pd
import logging
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "evinf", # sst
)

parser.add_argument(
    "--std",
    type = float,
    help = "std for noise distribution",
    default = 1,
)
parser.add_argument(
    "--std1",
    type = float,
    help = "std for noise distribution",
    default= None,
)

parser.add_argument(
    "--std2",
    type = float,
    help = "std for noise distribution",
    default= None,
)

parser.add_argument(
    "--std3",
    type = float,
    help = "std for noise distribution",
    default= None,
)

user_args = vars(parser.parse_args())
# user_args["importance_metric"] = None

dataset = str(user_args["dataset"])
pwd = os.getcwd()

topk_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'topk-faithfulness-scores-detailed.npy') 

NOISE_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'NOISElimit-faithfulness-scores-detailed-std_' + str(user_args["std"]) + '.npy') 
ATTENTION_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ATTENTIONlimit-faithfulness-scores-detailed.npy')
ZEROOUT_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ZEROOUTlimit-faithfulness-scores-detailed.npy')



TOPk_scores = np.load(topk_scores_file, allow_pickle=True).item()
ZEROOUT_scores = np.load(ZEROOUT_scores_file, allow_pickle=True).item()
ATTENTION_scores = np.load(ATTENTION_scores_file, allow_pickle=True).item()
NOISE_scores = np.load(NOISE_scores_file, allow_pickle=True).item()


data_id_list = TOPk_scores.keys()
fea_list = ['attention', "scaled attention", "gradients", "ig", "deeplift"] # "gradientshap",
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
        NOISE_random_suff_score = NOISE_scores.get(data_id).get('random').get('sufficiency aopc').get('mean')#.get('mean')
        ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get('sufficiency aopc').get('mean')#.get('mean')
        ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get('sufficiency aopc').get('mean')#.get('mean')

        top_suff_score = TOPk_scores.get(data_id).get(FA).get('sufficiency aopc').get('mean') # evinf @ 0.1
        if top_suff_score >= top_random_suff_score: Diag_TOP_attention += 1
        else: pass

        NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get('sufficiency aopc').get('mean')
        if NOISE_suff_score >= NOISE_random_suff_score: Diag_NOISE_attention += 1
        else: pass

        ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get('sufficiency aopc').get('mean')
        if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: Diag_ZEROOUT_attention += 1
        else: pass

        ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get('sufficiency aopc').get('mean')
        if ATTENTION_suff_score >= ATTENTION_random_suff_score: Diag_ATTENTION_attention += 1
        else: pass

        
    D_TOP = Diag_TOP_attention/len(data_id_list)
    D_TOP_Suff.append(D_TOP)

    D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
    D_ATTENTION_Suff.append(D_ATTENTION)

    D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
    D_ZEROOUT_Suff.append(D_ZEROOUT)

    D_NOISE= Diag_NOISE_attention/len(data_id_list)
    D_NOISE_Suff.append(D_NOISE)


df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ATTENTION_Suff, D_ZEROOUT_Suff, D_NOISE_Suff)),
               columns =['Feature', 'TopK', 'Soft(ATTENTION limit)', 'Soft(ZEROOUT limit)', 'Soft(NOISE limit)'])

fname = os.path.join(pwd, 'Diagnosticity', str(dataset), 'Diagnosticity_Suff_limit.csv')
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
        NOISE_random_suff_score = NOISE_scores.get(data_id).get('random').get('comprehensiveness aopc').get('mean')#.get('mean')
        ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get('comprehensiveness aopc').get('mean')#.get('mean')
        ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get('comprehensiveness aopc').get('mean')#.get('mean')

        top_suff_score = TOPk_scores.get(data_id).get(FA).get('comprehensiveness aopc').get('mean') # evinf @ 0.1
        if top_suff_score >= top_random_suff_score: Diag_TOP_attention += 1
        else: pass

        NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get('comprehensiveness aopc').get('mean')
        if NOISE_suff_score >= NOISE_random_suff_score: Diag_NOISE_attention += 1
        else: pass

        ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get('comprehensiveness aopc').get('mean')
        if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: Diag_ZEROOUT_attention += 1
        else: pass

        ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get('comprehensiveness aopc').get('mean')
        if ATTENTION_suff_score >= ATTENTION_random_suff_score: Diag_ATTENTION_attention += 1
        else: pass

    D_TOP = Diag_TOP_attention/len(data_id_list)
    D_TOP_Suff.append(D_TOP)

    D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
    D_ATTENTION_Suff.append(D_ATTENTION)

    D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
    D_ZEROOUT_Suff.append(D_ZEROOUT)

    D_NOISE= Diag_NOISE_attention/len(data_id_list)
    D_NOISE_Suff.append(D_NOISE)



df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ATTENTION_Suff, D_ZEROOUT_Suff, D_NOISE_Suff)),
               columns =['Feature', 'TopK', 'Soft(ATTENTION limit)', 'Soft(ZEROOUT limit)', 'Soft(NOISE limit)'])

fname = os.path.join(pwd, 'Diagnosticity', str(dataset), 'Diagnosticity_Comp-limit.csv')
os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)
df.to_csv(fname)



