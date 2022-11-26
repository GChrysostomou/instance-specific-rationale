import urllib.request
import os
import nltk
import spacy
import zipfile
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from cleaners_encoders import cleaner, tokenize, invert_and_join
import tarfile
import argparse
import csv
import sys
import shutil 
import pandas as pd
from sklearn.utils import shuffle
import gzip


csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_directory', 
    type=str, 
    help='directory to save processed data', 
    default = "datasets"
)

args = parser.parse_args()

AMAZ_DATA_ = {
    "AmazDigiMu": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music_5.json.gz",
    "AmazPantry" : "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry_5.json.gz",
    "AmazInstr" : "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments_5.json.gz"
}

import gc

TEMP_DATA_DIR="temp_data/"

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

def download_raw_data():
    ## create our temp data directory
    os.makedirs(TEMP_DATA_DIR, exist_ok = True)
    """
    Amazon datasets retrieved from Retrieved from 
    https://nijianmo.github.io/amazon/index.html#complete-data
    Justifying recommendations using distantly-labeled reviews and fined-grained aspects
    Jianmo Ni, Jiacheng Li, Julian McAuley
    Empirical Methods in Natural Language Processing (EMNLP), 2019
    """
    for dataset, url in AMAZ_DATA_.items():

        name = url.split("/")[-1]
        fname = os.path.join(
            os.getcwd(),
            TEMP_DATA_DIR,
            AMAZ_DATA_[task_name].split("/")[-1]
        ) # /home/cass/PycharmProjects/extract_rationales/temp_data/AmazDigiMu/

        #f"{TEMP_DATA_DIR}{name}" # eg ./src/data_functinos/temp_data/Digital_Music_5.json.gz
        print('current fname is :', fname)
        print(f"*** downloading and extracting data for {dataset}")
        if os.path.exists(fname):
            print(f"**** {TEMP_DATA_DIR}{name} allready exists")
            pass
        else:
            print(f"*** start downloading and extracting data for {dataset}")
            urllib.request.urlretrieve(
                url, 
                f"{TEMP_DATA_DIR}{name}")
    print("*** downloaded and extracted all temp files succesfully")
    return



class ProcessAmazonDatasets():
    """Processor for All Amazon datasets"""
    def load_samples_(self, task_name, path_to_data): ##  path_to_data eg. ./datasets/AmazDigiMu/data/
        print(' ============ start load ================', task_name)
        sent_lab = {2: "positive", 1: "neutral", 0:"negative"}
        data = []
        counter = 0
        print(' now processing: ', str(AMAZ_DATA_[task_name]))
        file_loc = os.path.join(
            os.getcwd(),
            TEMP_DATA_DIR,
            AMAZ_DATA_[task_name].split("/")[-1]
        )  ## e.g.  '/home/cass/PycharmProjects/extract_rationales/temp_data/Digital_Music_5.json.gz'
        with gzip.open(file_loc, "rb") as file: 

            for line in tqdm(file.readlines()):
                
                try:
                    
                    json_acceptable_string = line.decode("utf-8").replace("\"", "\"").rstrip()
                    d = json.loads(json_acceptable_string)
                    score = d["overall"]
                    date = d["reviewTime"]
                    verified = d["verified"]
                    label = self.score2sent(score)
                    data.append({
                        "text" : " ".join(cleaner(d["reviewText"])),
                        "true score" : score,
                        "label" : label,
                        "label_id" : sent_lab[label],
                        "date":  date,
                        "verified": verified
                    })

                except KeyError as e: 
                    
                    counter += 1
            
        print(f"*** failed to convert {counter} instances. This is due to KeyError (i.e. no review text found)")

        df = pd.DataFrame(data)

        with open(path_to_data + "fullset.json", "w") as file: ##  path_to_data eg. ./datasets/AmazDigiMu/data/
            json.dump(
                df.to_dict("records"),
                file,
                indent=4
            )

        train_indx, testdev_indx = train_test_split(df.index, test_size=0.2, stratify=df["label"])
        train = df.iloc[train_indx]
        testdev = df.loc[testdev_indx, :]
        train["split"] = "train"
        assert len([x for x in testdev.index if x in train.index]) == 0, ("""
        data leakage
        """)
        test_indx, dev_indx = train_test_split(testdev.index, test_size=0.5, stratify=testdev["label"])
        test = df.loc[test_indx, :]
        test["split"] = "test"
        dev = df.loc[dev_indx, :]
        dev["split"] = "dev"

        assert len([x for x in dev.index if x in test.index]) == 0, ("""
        data leakage
        """)

        train.reset_index(inplace = True)
        train["annotation_id"] = train.apply(lambda row: "train_" + str(row.name), axis = 1)

        dev.reset_index(inplace = True)
        dev["annotation_id"] = dev.apply(lambda row: "dev_" + str(row.name), axis = 1)

        test.reset_index(inplace = True)
        test["annotation_id"] = test.apply(lambda row: "test_" + str(row.name), axis = 1)


        # 存 xxx_full
        for split, data in {"train": train, "dev": dev, "test":test}.items():

            ## save our dataset
            full_data_directory = os.path.join(
                os.getcwd(),
                args.data_directory,
                task_name + '_full',
                "data",
                ""
            ) ##  ./datasets/AmazDigiMu_full/data/

            os.makedirs(full_data_directory, exist_ok=True)
            print('full_data_directory: ', str(full_data_directory))

            with open(full_data_directory + f"{split}.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4,

                )


        ### sorted dataset
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values(by='date', na_position='first')
        print(' full data oldest: ')
        print(df['date'][:5])
        print(' full data newst: ')
        print(df['date'][-5:])

        ood1_len = ood2_len = indomain_test_len = indomain_dev_len = int(len(df) * 0.1)
        in_domain_len = len(df) - ood1_len * 2
        assert len(df) == in_domain_len + ood2_len + ood1_len

        in_domain = df.iloc[:in_domain_len]
        in_domain = shuffle(in_domain)
        in_domain.reset_index(inplace=True)
        ood1 = df.iloc[in_domain_len:in_domain_len + ood1_len]
        ood2 = df.iloc[in_domain_len + ood1_len:]
        ood1.reset_index(inplace = True)
        ood1["annotation_id"] = ood1.apply(lambda row: "test_" + str(row.name), axis = 1)
        ood2.reset_index(inplace = True)
        ood2["annotation_id"] = ood2.apply(lambda row: "train_" + str(row.name), axis = 1)

        print('   ====   ood1 head')
        print(ood1['date'][:5])
        print('   ====   ood1 tail')
        print(ood1['date'][-5:])
        print('   ====   ood2 head')
        print(ood2['date'][-5:])
        print('   ====   ood1 tail')
        print(ood1['date'][-5:])

        in_domain_train_index, in_domain_test_index = train_test_split(in_domain.index, train_size=0.75)
        in_domain_train = in_domain.iloc[in_domain_train_index]
        in_domain_test = in_domain.loc[in_domain_test_index, :]
        in_domain_train["split"] = "train"

        in_domain_dev_index, in_domain_test_index = train_test_split(in_domain_test.index, train_size=0.5)
        in_domain_dev = in_domain_test.loc[in_domain_dev_index,:]
        in_domain_test = in_domain_test.loc[in_domain_test_index, :]
        in_domain_test["split"] = "test"
        in_domain_dev["split"] = "dev"

        in_domain_train.reset_index(inplace = True)
        in_domain_train["annotation_id"] = in_domain_train.apply(lambda row: "train_" + str(row.name), axis = 1)

        in_domain_dev.reset_index(inplace = True)
        in_domain_dev["annotation_id"] = in_domain_dev.apply(lambda row: "dev_" + str(row.name), axis = 1)

        in_domain_test.reset_index(inplace = True)
        in_domain_test["annotation_id"] = in_domain_test.apply(lambda row: "test_" + str(row.name), axis = 1)


        # 存 xxx
        for split, data in {"train": in_domain_train, "test": in_domain_test, "dev":in_domain_dev}.items():
            os.makedirs(path_to_data, exist_ok = True) ##  path_to_data eg. ./datasets/AmazDigiMu/data/
            print(split, 'is at : ', path_to_data)
            with open(path_to_data + f"{split}.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4,
                    default=str
                )
            print('saved train / test/ dev in domain at ', path_to_data)

        # 存 xxx_ood
        for split, data in {"ood1": ood1, "ood2": ood2}.items():
            ## save our dataset
            ood_data_directory = os.path.join(
                os.getcwd(),
                args.data_directory,
                task_name + '_' + str(split),
                "data",
                ""
            ) ### e.g. ./datasets/AmazDigiMu_ood1/

            os.makedirs(ood_data_directory, exist_ok=True)
            print('ood dataset directory: ', str(ood_data_directory))

            with open(ood_data_directory + "test.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4,
                    default=str
                )
            with open(ood_data_directory + "train.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4,
                    default=str
                )
            with open(ood_data_directory + "dev.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4,
                    default=str
                )
            print('saved ', str(split), 'at: ', ood_data_directory)
        print(' ============ done load ================', task_name)
        return


    def score2sent(self, score : float) -> int:
    
        if score > 3:
            return 2
        if score < 3:
            return 0    
        return 1


class AmazDigiMuProcessor(ProcessAmazonDatasets):

    def __init__(self):

        return

class AmazInstrProcessor(ProcessAmazonDatasets):

    def __init__(self):

        return

class AmazPantryProcessor(ProcessAmazonDatasets):

    def __init__(self):

        return



def describe_data_stats(path_to_data, path_to_stats):
    """ 
    returns dataset statistics such as : 
                                        - number of documens
                                        - average sequence length
                                        - average query length (if QA)
    """

    descriptions = {"train":{}, "dev":{}, "test":{}}
    
    for split_name in descriptions.keys():

        with open(f"{path_to_data}{split_name}.json", "r") as file: data = json.load(file) #  path_to_data eg. ./datasets/AmazDigiMu/data/

        if "query" in data[0].keys(): 


            avg_ctx_len = np.asarray([len(x["document"].split(" ")) for x in data]).mean()
            avg_query_len = np.asarray([len(x["query"].split(" ")) for x in data]).mean()

            descriptions[split_name]["avg. context length"] = int(avg_ctx_len)
            descriptions[split_name]["avg. query length"] = int(avg_query_len)

        else:

            avg_seq_len = np.asarray([len(x["text"].split(" ")) for x in data]).mean()

            descriptions[split_name]["avg. sequence length"] = int(avg_seq_len)

        descriptions[split_name]["no. of documents"] = int(len(data))
        
        label_nos = np.unique(np.asarray([x["label"] for x in data]), return_counts = True)

        for label, no_of_docs in zip(label_nos[0], label_nos[1]):

            descriptions[split_name][f"docs in label-{label}"] = int(no_of_docs)
    
    ## save descriptors
    fname = path_to_stats + "dataset_statistics.json"

    with open(fname, 'w') as file:
        
            json.dump(
                descriptions,
                file,
                indent = 4
            ) 


    del data
    del descriptions
    gc.collect()

    return


if __name__ == "__main__":
    ## download the raw temp data
    if os.path.isdir(TEMP_DATA_DIR):
        print(TEMP_DATA_DIR, ' already exist, no need download raw data again')
        pass
    else:
        print('starting download raw data')
        download_raw_data()

    ## processing raw data
    for task_name in {"AmazDigiMu", "AmazPantry", "AmazInstr"}: #"SST","IMDB", "Yelp",
        print(task_name)
        print(f"** processing -> {task_name}")

        processor = globals()[f'{task_name}Processor']()

        data_directory = os.path.join(
            os.getcwd(),
            args.data_directory,
            task_name, 
            "data",
            "", # eg. ./datasets/AmazDigiMu/data/
        )
        print('data_directory / path_to_data: ', data_directory)
        
        if "Amaz" in task_name:
            dataset = processor.load_samples_(
                task_name = task_name,
                path_to_data = data_directory,
            )
        else:
            dataset = processor.load_samples_(data_directory)

        ## save stats in 
        stats_directory = os.path.join(
            os.getcwd(),
            args.data_directory,
            task_name, 
            ""
        )

        describe_data_stats(
            path_to_data = data_directory,
            path_to_stats = stats_directory
        )

    print(f"** removing temporary data")

    # # deleting temp_data
    # shutil.rmtree(TEMP_DATA_DIR)
