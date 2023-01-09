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
    default = "sst", # sst agnews multirc
)




user_args = vars(parser.parse_args())
# user_args["importance_metric"] = None

dataset = str(user_args["dataset"])
pwd = os.getcwd()

topk_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'topk-faithfulness-scores-detailed.npy') 
# NOISE_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'NOISElimit-faithfulness-scores-detailed.npy') 
# ATTENTION_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ATTENTIONlimit-faithfulness-scores-detailed.npy')
ZEROOUT_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ZEROOUT-faithfulness-scores-detailed.npy')



TOPk_scores = np.load(topk_scores_file, allow_pickle=True).item()
ZEROOUT_scores = np.load(ZEROOUT_scores_file, allow_pickle=True).item()
# ATTENTION_scores = np.load(ATTENTION_scores_file, allow_pickle=True).item()
# NOISE_scores = np.load(NOISE_scores_file, allow_pickle=True).item() # key  feature_  suff/comp @



data_id_list = TOPk_scores.keys()
fea_list = ['attention', "scaled attention", "gradients", "ig", "deeplift"] # "gradientshap",
rationale_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]  #, 1.0

suff_or_comp = 'sufficiency' # sufficiency or comprehensiveness


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


            top_random_suff_score = TOPk_scores.get(data_id).get('random').get(f'{suff_or_comp} @ {str(ratio)}')#.get('mean')
            ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get(f'{suff_or_comp} @ 1.0')

            top_suff_score = TOPk_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ {str(ratio)}')
            if top_suff_score >= top_random_suff_score + 0.000001 : 
                Diag_TOP_attention += 1
                #ABS_Diag_TOP_attention += top_suff_score-top_random_suff_score
            else: pass


            ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ 1.0')
            
            if ZEROOUT_suff_score >= ZEROOUT_random_suff_score + 0.000001: 
                Diag_NOISE_attention += 1
                #ABS_Diag_NOISE_attention += NOISE_suff_score-NOISE_random_suff_score
            else: pass

            # ABS_Diag_TOP_attention += top_suff_score-top_random_suff_score
            # ABS_Diag_NOISE_attention += NOISE_suff_score-NOISE_random_suff_score
            # ABS_Diag_ZEROOUT_attention += ZEROOUT_suff_score-ZEROOUT_random_suff_score
            # ABS_Diag_ATTENTION_attention += ATTENTION_suff_score-ATTENTION_random_suff_score



        D_TOP = Diag_TOP_attention/len(data_id_list)
        D_TOP_Suff.append(D_TOP)

        D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
        D_ZEROOUT_Suff.append(D_ZEROOUT)



        ABS_D_TOP = ABS_Diag_TOP_attention/len(data_id_list)
        ABS_D_TOP_Suff.append(ABS_D_TOP)

        ABS_D_ZEROOUT = ABS_Diag_ZEROOUT_attention/len(data_id_list)
        ABS_D_ZEROOUT_Suff.append(ABS_D_ZEROOUT)


    if include_feature_name == True:
        df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ZEROOUT_Suff, D_NOISE_Suff, D_ATTENTION_Suff)),
                columns =['Feature', f'TopK {suff_or_comp} @ {str(ratio)}', f'Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', f'Soff(NOISE) {suff_or_comp} @ {str(ratio)}', f'Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}'])

        ABS_df = pd.DataFrame(list(zip(fea_list, ABS_D_TOP_Suff, ABS_D_ZEROOUT_Suff, ABS_D_NOISE_Suff, ABS_D_ATTENTION_Suff)),
                columns =['Feature', f'TopK {suff_or_comp} @ {str(ratio)}', f'Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', f'Soff(NOISE) {suff_or_comp} @ {str(ratio)}', f'Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}'])
    
    elif include_feature_name == False:
        df = pd.DataFrame(list(zip(D_TOP_Suff, D_ZEROOUT_Suff, D_NOISE_Suff, D_ATTENTION_Suff)),
                columns =[f'TopK {suff_or_comp} @ {str(ratio)}', f'Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', f'Soff(NOISE) {suff_or_comp} @ {str(ratio)}', f'Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}'])
        ABS_df = pd.DataFrame(list(zip(ABS_D_TOP_Suff, ABS_D_ATTENTION_Suff, ABS_D_ZEROOUT_Suff, ABS_D_NOISE_Suff)),
                columns =[f'TopK {suff_or_comp} @ {str(ratio)}', f'Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}', f'Soff(NOISE) {suff_or_comp} @ {str(ratio)}', f'Soff(ATTENTION) {suff_or_comp} @ {str(ratio)}'])
        
    else:print('need to define if include baseline or not')


    return df, ABS_df


rationale_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] 
os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)


for ratio in rationale_ratios:

    to_be_concatenated_df, ABS_to_be_concatenated_df = generate_table(suff_or_comp, ratio, False)
    print(' ')
    print(' ')
    print(' ')
    print(f' {ratio} ---> Diagnosticity and ABS Diagnosticity')
    print(' ')
    print(to_be_concatenated_df)
    print(ABS_to_be_concatenated_df)

    if ABS_to_be_concatenated_df[f'Soff(ZEROOUT) {suff_or_comp} @ {str(ratio)}'].sum() >= ABS_to_be_concatenated_df[f'TopK {suff_or_comp} @ {str(ratio)}'].sum():


        fname = os.path.join(pwd, 'Diagnosticity', str(dataset), f'D_{suff_or_comp}_{ratio}.csv')
        ABS_fname = os.path.join(pwd, 'Diagnosticity', str(dataset), f'ABS_D_{suff_or_comp}_{ratio}.csv')

        to_be_concatenated_df.to_csv(fname)
        ABS_to_be_concatenated_df.to_csv(ABS_fname)
        print(' ')
        print(' ')
        print(' ------------------- ')
        print(F'DONE {suff_or_comp} @ {str(ratio)}')
        print(' ------------------- ')































def generate_aopc(detailed_dict, suff_or_comp, ratio_list):
    D_TOP_Suff = []
    D_ATTENTION_Suff = []
    D_ZEROOUT_Suff = []
    D_NOISE_Suff = []

    FA_AOPC_list = []
    for FA in fea_list:


        aopc_list_for_all_data = 0

        for i, data_id in enumerate(data_id_list):
            score_at_diff_ratio = 0

            for ratio in ratio_list:
                score_at_one_ratio = TOPk_scores.get(data_id).get(FA).get(f'{suff_or_comp} @ {str(ratio)}')#.get('mean')
                score_at_diff_ratio += score_at_one_ratio
            aopc_one_data = score_at_diff_ratio/len(ratio_list)
        
        aopc_list_for_all_data += aopc_one_data

    FA_AOPC_list.append(aopc_list_for_all_data/len(fea_list))


    df = pd.DataFrame(list(zip(fea_list, FA_AOPC_list)),
                columns =['Feature', 'AOPC'])

    return df




