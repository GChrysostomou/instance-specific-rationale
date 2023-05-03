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


import pickle 


        
with open('results_summary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)


def get_dict(dataset, model_folder_name, model_abb):        
        model_folder = model_folder_name + '_trained_models/' + dataset + '/'


        file_path = f'./{model_folder_name}_faithfulness/{dataset}/topk-faithfulness-scores-average-description.json'
        topk = pd.read_json(file_path, orient ='index')
        topk.rename(columns = {'AOPC - sufficiency':'AOPC_sufficiency', 'AOPC - comprehensiveness':'AOPC_comprehensiveness'}, inplace = True)

        file_path = f'./{model_folder}{model_abb}_predictive_performances.json'
        pred = pd.read_json(file_path)
        if dataset == 'sst' or dataset == 'evinf' or dataset == 'multirc' or dataset == 'agnews': pred_result = pred['mean-f1'].mean() 
        else: pred_result = pred['mean-accuracy'].mean()

        suff = []
        comp = []
        suff_std = []
        comp_std = []
        fea_list = ['random', 'attention', "scaled attention", "gradients", "ig", "deeplift"] #"gradientshap", 
        for feat in fea_list:
                suff.append(topk.AOPC_sufficiency[str(feat)].get('mean'))
                comp.append(topk.AOPC_comprehensiveness[str(feat)].get('mean'))
                
                suff_std.append(topk.AOPC_sufficiency[str(feat)].get('std'))
                comp_std.append(topk.AOPC_comprehensiveness[str(feat)].get('std'))
        fea_list = ['Random', 'Attention', "Scaled Attention", "Gradients", "Integrated Gradients", "Deeplift"]
        final_df = pd.DataFrame(list(zip(fea_list, suff, suff_std, comp, comp_std)),
                columns =['Feature','AOPC NormSuff', 'std', 'AOPC NormComp', 'std'])
        final_df['Dataset'] = str(dataset)

        df = final_df.T
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
        df = df[:-1] # drop the las dataset name row

        df['Attention'] = df['Attention']/df['Random']
        df['Scaled Attention'] = df['Scaled Attention']/df['Random']
        df['Gradients'] = df['Gradients']/df['Random']
        df['Integrated Gradients'] = df['Integrated Gradients']/df['Random']
        df['Deeplift'] = df['Deeplift']/df['Random']

        pred_faith_dict = {}
        pred_faith_dict['Attention_Suff'] = df['Attention']['AOPC NormSuff']
        pred_faith_dict['Scaled_Attention_Suff'] = df['Scaled Attention']['AOPC NormSuff']
        pred_faith_dict['Gradients_Suff'] = df['Gradients']['AOPC NormSuff']
        pred_faith_dict['Integrated_Gradients_Suff'] = df['Integrated Gradients']['AOPC NormSuff']
        pred_faith_dict['Deeplift_Suff'] = df['Deeplift']['AOPC NormSuff']
        pred_faith_dict['Attention_Comp'] = df['Attention']['AOPC NormComp']
        pred_faith_dict['Scaled_Attention_Comp'] = df['Scaled Attention']['AOPC NormComp']
        pred_faith_dict['Gradients_Comp'] = df['Gradients']['AOPC NormComp']
        pred_faith_dict['Integrated_Gradients_Comp'] = df['Integrated Gradients']['AOPC NormComp']
        pred_faith_dict['Deeplift_Comp'] = df['Deeplift']['AOPC NormComp']


        if dataset == 'ChnSentiCorp' or dataset == 'ant' or dataset == 'csl':
            pred_faith_dict['Accuracy'] = pred_result
        else: pred_faith_dict['F1'] = pred_result


        model_pred_faith_dict = {}
        model_pred_faith_dict[model_abb] = pred_faith_dict
        return model_pred_faith_dict


#ChnSentiCorp_mbert_dict = get_dict('ChnSentiCorp','mbert')


# mbert m /  
model_folder_name = 'french_roberta' #french_roberta BETO
model_abb = 'french_roberta'
data = 'french_csl'  # spanish_csl french_paws french_csl french_xnli 
                      # ChnSentiCorp ant csl multirc agnews sst evinf
current_data_model_dict_noDATAhead = get_dict(data,model_folder_name, model_abb)

dataset_list = loaded_dict.keys()


print(' ')
if data in dataset_list: 
    print(f' have the data {data} in dict already !!!')
    model_list = loaded_dict[data].keys()
    print(' ONLY update model --->', model_abb)
    loaded_dict[data].update(current_data_model_dict_noDATAhead)
else: 
     print(f' add {data} and {model_abb} to the dict')
     loaded_dict[data] = current_data_model_dict_noDATAhead




print(' ')

for data in loaded_dict.keys():
     print(' ')
     print(data)
     print('------')
     for model in loaded_dict[data].keys():
          print(model)
          #print(loaded_dict[data][model])


with open('results_summary.pkl', 'wb') as f:
    pickle.dump(loaded_dict, f)

quit()

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


