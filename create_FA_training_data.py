import numpy as np
import pandas as pd
import torch
import os 
import argparse
import logging
from sklearn.model_selection import train_test_split


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







def feature2label(split, original_data, meta_result_data_path, saved_data_path):
    metadata = np.load(meta_result_data_path, allow_pickle=True).item(0)
    id_list = list(metadata.keys())
    best_feat_list = []
    for id in id_list:
        best_feat = metadata[id]['var-len_var-feat_var-type']['feature attribution name']
        best_feat_list.append(best_feat)


    df = pd.DataFrame(list(zip(id_list, best_feat_list)), columns =['annotation_id', 'feat'])
    original_data = pd.read_csv(original_data)
    FAdata = pd.merge(original_data, df, on="annotation_id")
    FAdata['label_id'] = FAdata['feat'] # change original label to features 
    FAdata['label'] = pd.factorize(FAdata['label_id'])[0]
    print(FAdata['label'])
    FAdata.to_csv(saved_data_path + str(split) +'.csv')
    return FAdata



spl = ['test', 'dev']
for split in spl:

    original_data = original_data_path + str(split) + '.csv'
    saved_data_path_top = user_args['data_dir'] + user_args['dataset'] + '_top/data/'
    saved_data_path_conti = user_args['data_dir'] + user_args['dataset'] + '_conti/data/'
    os.makedirs(saved_data_path_top, exist_ok = True)
    os.makedirs(saved_data_path_conti, exist_ok = True)
    meta_result_data_path_top = 'extracted_rationales/' + user_args['dataset'] + '/topk/'+ str(split) +'-rationale_metadata.npy'
    meta_result_data_path_conti = 'extracted_rationales/' + user_args['dataset'] + '/contigious/'+ str(split) +'-rationale_metadata.npy'

    test_top = feature2label(str(split), original_data, meta_result_data_path_top, saved_data_path_top)
    test_conti = feature2label(str(split), original_data, meta_result_data_path_conti, saved_data_path_conti)



train, dev = train_test_split(test_top, test_size=0.2, random_state=412)
train.to_csv(saved_data_path_top + 'train.csv')
dev.to_csv(saved_data_path_top + 'dev.csv')

train, dev = train_test_split(test_conti, test_size=0.2, random_state=412)
train.to_csv(saved_data_path_conti + 'train.csv')
dev.to_csv(saved_data_path_conti + 'dev.csv')

print('train: ',len(train), ' test:', len(test_conti), ' dev:', len(dev))