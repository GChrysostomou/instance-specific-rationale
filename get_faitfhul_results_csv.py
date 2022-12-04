from pickle import NONE
from re import T
import pandas as pd
import json
import glob
import os 
import argparse
import logging



import datetime
import gc

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst", 
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

def generate_csv(dataset, method, std, path):
    file_path = faithful_result + dataset + '/' + path
    print(file_path)

    df = pd.read_json(file_path, orient ='index')
    #print(df)
    df.rename(columns = 
            {'AOPC - sufficiency':'AOPC_sufficiency', 'AOPC - comprehensiveness':'AOPC_comprehensiveness'}, 
            inplace = True)
    sufficiency_mean = []
    comprehensiveness_mean = []


    fea_list = ['random', 'attention', "scaled attention", "gradients", "ig", "deeplift"] #"gradientshap", 
    for feat in fea_list:
        if method == 'topk':
            sufficiency_mean.append(df.AOPC_sufficiency[str(feat)].get('mean'))
            comprehensiveness_mean.append(df.AOPC_comprehensiveness[str(feat)].get('mean'))
        else: # soft true --> no aopc
            sufficiency_mean.append(df.sufficiency[str(feat)].get('mean'))
            comprehensiveness_mean.append(df.comprehensiveness[str(feat)].get('mean'))



    if method != 'topk':
        random_suff = df.sufficiency['random'].get('mean')
        random_comp = df.comprehensiveness['random'].get('mean')
        
        Suff_ratio = [x / random_suff for x in sufficiency_mean]
        Comp_ratio = [x / random_comp for x in comprehensiveness_mean]

        final_df = pd.DataFrame(list(zip(fea_list, sufficiency_mean, Suff_ratio, comprehensiveness_mean, Comp_ratio)),
                columns =['feature', 'Soft_sufficiency', 'Suff_ratio', 'Soft_comprehensiveness', 'Comp_ratio'])
        
    
        if 'NOISE' in method: final_path = faithful_result + dataset + '/' + str(method) + str(std) +'_faithfulness_result.csv'
        else: final_path = faithful_result + dataset + '/' + str(method) +'_faithfulness_result.csv'
        
        final_df.to_csv(final_path)
        print('saved csv: ', final_path)

    else: # not soft, so have aopc
        random_suff = df.AOPC_sufficiency['random'].get('mean')
        random_comp = df.AOPC_comprehensiveness['random'].get('mean')

        Suff_ratio = [x / random_suff for x in sufficiency_mean]
        Comp_ratio = [x / random_comp for x in comprehensiveness_mean]

        final_df = pd.DataFrame(list(zip(fea_list, sufficiency_mean, Suff_ratio, comprehensiveness_mean, Comp_ratio)),
                columns =['feature', ' AOPC_sufficiency', 'Suff_ratio', 'AOPC_comprehensiveness', 'Comp_ratio'])
        final_path = faithful_result + dataset + '/' +'faithfulness_result.csv'
        print('saved csv: ', final_path)
        final_df.to_csv(final_path)


# try: generate_csv(str(dataset), 'NOISE', 1, 'NOISE-faithfulness-scores-description-std_1.json')
# except: generate_csv(str(dataset), 'NOISE', 1, 'NOISE-faithfulness-scores-description-std_1.0.json')



#generate_csv(str(dataset), 'topk', 1, 'topk-faithfulness-scores-average-description.json')

# generate_csv(str(dataset), 'ATTENTION', 1, 'ATTENTION-faithfulness-scores-description.json')
# generate_csv(str(dataset), 'ATTENTIONlimit', 1, 'ATTENTIONlimit-faithfulness-scores-description.json')

# generate_csv('agnews', 'ATTENTION', 1, 'ATTENTION-faithfulness-scores-description.json')
generate_csv('sst', 'ATTENTIONlimit', 1, 'ATTENTIONlimit-faithfulness-scores-description.json')

#generate_csv('agnews', 'ZEROOUT', 1, 'ZEROOUT-faithfulness-scores-description.json')
#generate_csv('agnews', 'ZEROOUTlimit', 1, 'ZEROOUTlimit-faithfulness-scores-description.json')

# generate_csv(str(dataset), 'NOISE', 0.5, 'NOISE-faithfulness-scores-description-std_0.5.json')
# generate_csv(str(dataset), 'NOISElimit', 0.5, 'NOISElimit-faithfulness-scores-description-std_0.5.json')






