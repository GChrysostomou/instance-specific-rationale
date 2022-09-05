import numpy as np
import pandas as pd
import torch
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
    default = "evinf", 
    #choices = ["sst", "evinf", "agnews", "multirc"]
)

parser.add_argument(
    "--data_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)

parser.add_argument(
    "--FA_data_dir", 
    type = str, 
    help = "directory for saving created FA data", 
    default = "FA_datasets/"
)

parser.add_argument(
    "--model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "trained_models/"
)

parser.add_argument(
    "--seed",   
    type = int, 
    help = "random seed for experiment"
)

# parser.add_argument(
#     '--evaluate_models', 
#     help='test predictive performance in and out of domain', 
#     action='store_true',
#     default=True,
# )

user_args = vars(parser.parse_args())


### used only for data stats
data_dir_plain = user_args["data_dir"]


log_dir = "experiment_logs/createFAdata_" + user_args["dataset"] + "_seed-" +  date_time + "/"
config_dir = "experiment_config/createFAdata_" + user_args["dataset"] + "_seed-"  + date_time + "/"


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

logging.info("Running on cuda : {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from src.common_code.initialiser import initial_preparations
import datetime

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "train")

logging.info("creating FA data for ")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")


original_data_path = user_args['data_dir'] + user_args['dataset'] + '/data/'

saved_data_path = user_args['data_dir'] + user_args['dataset'] + '_FA/data/'
os.makedirs(saved_data_path, exist_ok = True)


original_data = original_data_path + 'test.csv'
meta_result_data_path = 'extracted_rationales/' + user_args['dataset'] + '/topk/test-rationale_metadata.npy'

metadata = np.load(meta_result_data_path, allow_pickle=True).item(0)
id_list = list(metadata.keys())
best_feat_list = []
for id in id_list:
    best_feat = metadata[id]['var-len_var-feat_var-type']['feature attribution name']
    best_feat_list.append(best_feat)

import collections
freq_rank = collections.Counter(best_feat_list).most_common()

common_selected_FA = []
for FA in freq_rank:
    FA_ratio = FA[1]/len(best_feat_list)
    if FA_ratio > 0.1:
        common_selected_FA.append(FA[0])

print('common_selected_FA:  ', common_selected_FA)


df = pd.DataFrame(list(zip(id_list, best_feat_list)),
            columns =['annotation_id', 'feat'])
original_data = pd.read_csv(original_data)

FAdata = pd.merge(original_data, df, on="annotation_id")
FAdata['label_id'] = best_feat_list # change original label to features 


converted_label_id_list = []
print(FAdata['label_id'])

for label in FAdata['label_id']:
    if label not in common_selected_FA:
        converted_label_id_list.append('others')
    else:
        converted_label_id_list.append(label)

FAdata['label_id'] = converted_label_id_list
print(collections.Counter(converted_label_id_list).most_common())

#df.loc[df['Population(B)'] < 1] = 0
FAdata['label'] = pd.factorize(FAdata['label_id'])[0]
print(FAdata['label'])

FAdata.to_csv(saved_data_path + 'full_FAdata.csv')
print(' ========= full test saved at: ', saved_data_path + 'full_FAdata.csv')
print('length ', len(FAdata))




# from src.data_functions.dataholder import classification_dataholder as dataholder
# from src.tRpipeline import train_and_save, test_predictive_performance, keep_best_model_
# from src.data_functions.useful_functions import describe_data_stats
