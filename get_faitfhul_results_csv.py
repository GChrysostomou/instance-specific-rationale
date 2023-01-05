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




def generate_csv(dataset, method, normal, std):
    file_path = f'./posthoc_results/{dataset}/topk-faithfulness-scores-average-description.json'
    topk = pd.read_json(file_path, orient ='index')
    topk.rename(columns = {'AOPC - sufficiency':'AOPC_sufficiency', 'AOPC - comprehensiveness':'AOPC_comprehensiveness'}, inplace = True)

    file_path = f'./posthoc_results/{dataset}/{method}-faithfulness-scores-normal_{normal}.json'
    soft = pd.read_json(file_path, orient ='index')
    soft.rename(columns = {'sufficiencies @ 1.0':'SoftSuff', 'comprehensiveness @ 1.0':'SoftComp'}, inplace = True)
    
    suff = []
    comp = []
    soft_suff = []
    soft_comp = []
    suff_std = []
    comp_std = []
    soft_suff_std = []
    soft_comp_std = []


    fea_list = ['random', 'attention', "scaled attention", "gradients", "ig", "deeplift"] #"gradientshap", 
    for feat in fea_list:
        suff.append(topk.AOPC_sufficiency[str(feat)].get('mean'))
        comp.append(topk.AOPC_comprehensiveness[str(feat)].get('mean'))
        soft_suff.append(soft.SoftSuff[str(feat)].get('mean'))
        soft_comp.append(soft.SoftComp[str(feat)].get('mean'))

        suff_std.append(topk.AOPC_sufficiency[str(feat)].get('std'))
        comp_std.append(topk.AOPC_comprehensiveness[str(feat)].get('std'))
        soft_suff_std.append(soft.SoftSuff[str(feat)].get('std'))
        soft_comp_std.append(soft.SoftComp[str(feat)].get('std'))


    # if method != 'topk':
    #     random_suff = df.sufficiency['random'].get('mean')
    #     random_comp = df.comprehensiveness['random'].get('mean')
        
    #     Suff_ratio = [x / random_suff for x in sufficiency_mean]
    #     Comp_ratio = [x / random_comp for x in comprehensiveness_mean]

    #     final_df = pd.DataFrame(list(zip(fea_list, sufficiency_mean, Suff_ratio, comprehensiveness_mean, Comp_ratio)),
    #             columns =['Feature', 'Soft-NormSuff', 'Suff_ratio', 'Soft_comprehensiveness', 'Comp_ratio'])
        
    
    #     if 'NOISE' in method: final_path = faithful_result + dataset + '/' + str(method) + str(std) +'_faithfulness_result.csv'
    #     else: final_path = faithful_result + dataset + '/' + str(method) +'_faithfulness_result.csv'
        
    #     final_df.to_csv(final_path)
    #     print('saved csv: ', final_path)

    # else: # not soft, so have aopc
    #     random_suff = df.AOPC_sufficiency['random'].get('mean')
    #     random_comp = df.AOPC_comprehensiveness['random'].get('mean')

    #     Suff_ratio = [x / random_suff for x in sufficiency_mean]
    #     Comp_ratio = [x / random_comp for x in comprehensiveness_mean]
    fea_list = ['Random', 'Attention', "Scaled Attention", "Gradients", "Integrated Gradients", "Deeplift"]

    final_df = pd.DataFrame(list(zip(fea_list, soft_suff, soft_suff_std, soft_comp, soft_comp_std, 
                                        suff, suff_std, comp, comp_std)),
            columns =['Feature', 'Soft-NormSuff', 'std', 'Soft-NormComp', 'std', 
                                ' AOPC NormSuff', 'std', 'AOPC NormComp', 'std'])
    final_df['Dataset'] = str(dataset)


    return final_df


df1 = generate_csv(dataset='sst', method='ZEROOUT', normal=1, std=0.5)
df2 = generate_csv(dataset='agnews', method='ZEROOUT', normal=1, std=0.5)
#df3 = generate_csv(dataset='multirc', method='ZEROOUT', normal=1, std=0.5)
#df4 = generate_csv(dataset='evinf', method='ZEROOUT', normal=1, std=0.5)

final_df = pd.concat([df1, df2])
final_path = './posthoc_results/faithfulness_result.csv'
print('saved csv: ', final_path)
final_df.to_csv(final_path)


