from pickle import NONE
from re import T
import pandas as pd
import json
import glob
import os 
import argparse
import logging
import numpy as np


import datetime
import gc

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "agnews", 
    #choices = ["sst", "evinf", "agnews", "multirc"]
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
    help = "directory to save faithfulness results", 
    default = "posthoc_results/"
)

user_args = vars(parser.parse_args())


faithful_result = user_args['evaluation_dir']
dataset = user_args['dataset']



pwd = os.getcwd()
topk_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'topk-faithfulness-scores-detailed.npy') 
NOISE_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'NOISElimit-faithfulness-scores-detailed.npy') 
ATTENTION_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ATTENTIONlimit-faithfulness-scores-detailed.npy')
ZEROOUT_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ZEROOUTlimit-faithfulness-scores-detailed.npy')


TOPk_scores = np.load(topk_scores_file, allow_pickle=True).item()
ZEROOUT_scores = np.load(ZEROOUT_scores_file, allow_pickle=True).item()
ATTENTION_scores = np.load(ATTENTION_scores_file, allow_pickle=True).item()
NOISE_scores = np.load(NOISE_scores_file, allow_pickle=True).item() # key  feature_  suff/comp @


data_id_list = TOPk_scores.keys()
fea_list = ['attention', "scaled attention", "gradients", "ig", "deeplift"] # "gradientshap",
rationale_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] 

suff_or_comp = 'comprehensiveness' # sufficiencies or comprehensiveness


def open_file(file_path):
    topk_faith = pd.read_json(file_path, orient ='index')
    topk_faith.rename(columns = {'AOPC - sufficiency':'AOPC_sufficiency', 'AOPC - comprehensiveness':'AOPC_comprehensiveness'}, inplace = True)
    return topk_faith

def generate_table(suff_or_comp, ratio, include_feature_name=True):
    D_TOP_Suff = []
    D_ATTENTION_Suff = []
    D_ZEROOUT_Suff = []
    D_NOISE_Suff = []

    ABS_D_TOP_Suff = []
    ABS_D_ATTENTION_Suff = []
    ABS_D_ZEROOUT_Suff = []
    ABS_D_NOISE_Suff = []

    for FA in fea_list:
        Diag_TOP_attention = 0
        Diag_ATTENTION_attention = 0
        Diag_ZEROOUT_attention = 0
        Diag_NOISE_attention = 0

        ABS_Diag_TOP_attention = 0
        ABS_Diag_ATTENTION_attention = 0
        ABS_Diag_ZEROOUT_attention = 0
        ABS_Diag_NOISE_attention = 0
        for i, data_id in enumerate(data_id_list):
            top_random_suff_score = TOPk_scores.get(data_id).get('random').get(f'{suff_or_comp} @ {str(ratio)}')
            NOISE_random_suff_score = NOISE_scores.get(data_id).get('random').get(f'{suff_or_comp} @ {str(ratio)}')
            ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get(f'{suff_or_comp} @ {str(ratio)}')
            ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get(f'{suff_or_comp} @ {str(ratio)}')

            top_suff_score = TOPk_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ {str(ratio)}')
            if top_suff_score >= top_random_suff_score: 
                Diag_TOP_attention += 1
                ABS_Diag_TOP_attention += top_suff_score
            else: pass

            NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ {str(ratio)}')
            if NOISE_suff_score >= NOISE_random_suff_score: 
                Diag_NOISE_attention += 1
                ABS_Diag_ATTENTION_attention += NOISE_suff_score
            else: pass

            ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ {str(ratio)}')
            if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: 
                Diag_ZEROOUT_attention += 1
                ABS_Diag_ZEROOUT_attention += ZEROOUT_suff_score
            else: pass

            ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ {str(ratio)}')
            if ATTENTION_suff_score >= ATTENTION_random_suff_score: 
                Diag_ATTENTION_attention += 1
                ABS_Diag_NOISE_attention += ATTENTION_suff_score
            else: pass

            
        D_TOP = Diag_TOP_attention/len(data_id_list)
        D_TOP_Suff.append(D_TOP)
        ABS_D_TOP = ABS_Diag_TOP_attention/len(data_id_list)
        ABS_D_TOP_Suff.append(ABS_D_TOP)

        D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
        D_ATTENTION_Suff.append(D_ATTENTION)
        ABS_D_ATTENTION = ABS_Diag_ATTENTION_attention/len(data_id_list)
        ABS_D_ATTENTION_Suff.append(ABS_D_ATTENTION)

        D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
        D_ZEROOUT_Suff.append(D_ZEROOUT)
        ABS_D_ZEROOUT = ABS_Diag_ZEROOUT_attention/len(data_id_list)
        ABS_D_ZEROOUT_Suff.append(ABS_D_ZEROOUT)

        D_NOISE= Diag_NOISE_attention/len(data_id_list)
        D_NOISE_Suff.append(D_NOISE)
        ABS_D_NOISE= ABS_Diag_NOISE_attention/len(data_id_list)
        ABS_D_NOISE_Suff.append(ABS_D_NOISE)



    ## get faith scores from description json

    topk_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'topk-faithfulness-scores-average-description.json') 
    NOISE_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'NOISElimit-faithfulness-scores-description.json') 
    ATTENTION_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ATTENTIONlimit-faithfulness-scores-description.json')
    ZEROOUT_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ZEROOUTlimit-faithfulness-scores-description.json')

    topk_df = open_file(topk_scores_file)
    print(topk_df)
    noise_df = open_file(NOISE_scores_file)
    attention_df = open_file(ATTENTION_scores_file)
    zeroout_df = open_file(ZEROOUT_scores_file)

    topk_suff_or_comp_mean = []
    noise_suff_or_comp_mean = []
    attention_suff_or_comp_mean = []
    zeroout_suff_or_comp_mean = []

    if suff_or_comp == 'sufficiency': suff_or_comp = 'sufficiencies'
    else: pass


    topk_random_faitfhul_score = topk_df.loc['random'][f'{suff_or_comp} @ {ratio}'].get('mean')
    noise_random_faitfhul_score = noise_df.loc['random'][f'{suff_or_comp} @ {ratio}'].get('mean')
    attention_random_faitfhul_score = attention_df.loc['random'][f'{suff_or_comp} @ {ratio}'].get('mean')
    zeroout_random_faitfhul_score = zeroout_df.loc['random'][f'{suff_or_comp} @ {ratio}'].get('mean')
    #print(noise_random_faitfhul_score, attention_random_faitfhul_score, zeroout_random_faitfhul_score)
    
    for FA in fea_list:
        
        # print(topk_df.loc[FA][f'{suff_or_comp} @ {ratio}'].get('mean'))
        # print(topk_random_faitfhul_score)

        topk_suff_or_comp_mean.append((topk_df.loc[FA][f'{suff_or_comp} @ {ratio}'].get('mean'))/(topk_random_faitfhul_score))
        #print(noise_df.loc[FA][f'{suff_or_comp} @ {ratio}'].get('mean'))
        noise_suff_or_comp_mean.append((noise_df.loc[FA][f'{suff_or_comp} @ {ratio}'].get('mean'))/(noise_random_faitfhul_score))
        attention_suff_or_comp_mean.append((attention_df.loc[FA][f'{suff_or_comp} @ {ratio}'].get('mean'))/(attention_random_faitfhul_score))
        zeroout_suff_or_comp_mean.append((zeroout_df.loc[FA][f'{suff_or_comp} @ {ratio}'].get('mean'))/(zeroout_random_faitfhul_score))
    


    final_big_df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, ABS_D_TOP_Suff, topk_suff_or_comp_mean, 
                                D_ZEROOUT_Suff, ABS_D_ZEROOUT_Suff, zeroout_suff_or_comp_mean,
                                D_NOISE_Suff,  ABS_D_NOISE_Suff, noise_suff_or_comp_mean,
                                D_ATTENTION_Suff, ABS_D_ATTENTION_Suff, attention_suff_or_comp_mean)),
                columns =['Feature', f'D - TopK {suff_or_comp} @ {str(ratio)}', f'ABS D - TopK {suff_or_comp} @ {str(ratio)}', f'TopK {suff_or_comp} @ {str(ratio)}', 
                f'D - Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', f'ABS D - Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', f'Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', 
                f'D - Soff(NOISE) {suff_or_comp} @ {str(ratio)}', f'ABS D - Soff(NOISE) {suff_or_comp} @ {str(ratio)}', f'Soff(NOISE) {suff_or_comp} @ {str(ratio)}', 
                f'D - Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}', f'ABS D - Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}', f'Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}'])


    fname = os.path.join(pwd, 'Diagnosticity', str(dataset), f'Soft_{suff_or_comp}.csv')
    os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)
    final_big_df.to_csv(fname)
    #return final_big_df



generate_table(suff_or_comp, 0.2, include_feature_name=True)