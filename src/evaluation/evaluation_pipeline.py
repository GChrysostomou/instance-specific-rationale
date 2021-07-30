
import torch
import pandas as pd
import json 
import glob 
import os
import numpy as np
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.bert import bert
from src.variable_rationales.var_length import get_rationale_metadata_
from src.evaluation.experiments.rationale_extractor import rationale_creator_
from src.evaluation.experiments.erasure_tests import conduct_tests_
from src.evaluation.experiments.increasing_feature_scoring import compute_faithfulness_

import re

from sklearn.metrics import classification_report

class evaluate():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, model_path, output_dims = 2):
        
        """
        loads and holds a pretrained model
        """

        self.models = glob.glob(model_path + args["model_abbreviation"] + "*.pt")
        self.output_dims = output_dims

        if len(self.models) < 1:

            raise OSError(f"model list is empty at -> {model_path} \n make sure you have the correct model path") from None

        logging.info(f" *** there are {len(self.models)} models in :  {model_path}")


    def prepare_for_rationale_creation_(self,data):

        for model_name in self.models:

            model = bert(
                masked_list=[0,101,102],
                output_dim = self.output_dims
            )


            logging.info(f" *** loading model -> {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model -> {model_name}")

            self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## train neglected as we are evaluating on dev and test
            for data_split_name, data_split in {"dev":  data.dev_loader , \
                                                "test":  data.test_loader}.items():

                fname = os.path.join(
                    os.getcwd(),
                    args["extracted_rationale_dir"],
                    args["thresholder"],
                    data_split_name + "-rationale_metadata.npy"
                )

                if os.path.isfile(fname):

                    print(f"rationale metadata file exists at {fname}") 
                    print("remove if you would like to rerun")

                    continue 

                get_rationale_metadata_(
                    model = model, 
                    data_split_name = data_split_name,
                    data = data_split,
                    model_random_seed = self.model_random_seed
                )


        return

    def create_rationales_(self, data):
        
        for data_split_name, data_split in data.as_dataframes_().items():

            
            rationale_creator_(
                data = data_split,
                data_split_name = data_split_name,
                tokenizer = data.tokenizer,
                variable = False
            )

            rationale_creator_(
                data = data_split,
                data_split_name = data_split_name,
                tokenizer = data.tokenizer,
                variable = True
            )



        return

    def faithfulness_metrics_(self, data):
        
        for model_name in self.models:
            
            ## check first if necessary data exists
            fname = os.path.join(
                os.getcwd(),
                args["extracted_rationale_dir"],
                args["thresholder"],
                "test-rationale_metadata.npy"
            )

            if os.path.isfile(fname) == False:

                raise OSError(f"rationale metadata file does not exist at {fname} // rerun extract_rationales.py") from None
          
            model = bert(
                                masked_list=[0,101,102],
                                output_dim = self.output_dims
                            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            conduct_tests_(
                model = model, 
                data = data.test_loader,
                model_random_seed = model_random_seed
            )

        return


    def feature_scoring_performance_(self):

        ## load rationale metadata for divergence scores
        ## WARNING // dependent on computing all the rationale metadata
        fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            args["thresholder"],
            "test-rationale_metadata.npy"
        )

        if os.path.isfile(fname) == False:

                raise OSError(f"rationale metadata file does not exist at {fname} // rerun extract_rationales.py") from None
        

        ## retrieve importance scores
        rationale_metadata = np.load(fname, allow_pickle = True).item()

        ## load now for the predictions
        ## WARNING // dependent on running the first set of experiment for evaluating masked rationales
        fname = os.path.join(
            os.getcwd(),
            args["evaluation_dir"],
            args["thresholder"] + "-faithfulness-metrics.json"
        )


        if os.path.isfile(fname) == False:

            raise OSError(f"faithfulness metrics file does not exist at {fname} // rerun experiments in this file") from None
        
        ## retrieve predictions
        with open(fname, "r") as file : prediction_data = json.load(file)

        compute_faithfulness_(
            rationale_metadata=rationale_metadata,
            prediction_data=prediction_data
        )

        
        return


    def token_wise_performance(self):

        ## load rationale metadata for divergence scores
        ## WARNING // dependent on computing all the rationale metadata
        fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            args["thresholder"],
            "test-rationale_metadata.npy"
        )

        if os.path.isfile(fname) == False:

                raise OSError(f"rationale metadata file does not exist at {fname} // rerun extract_rationales.py") from None
        

        ## retrieve importance scores
        rationale_metadata = np.load(fname, allow_pickle = True).item()

        ## load now for the predictions
        ## WARNING // dependent on running the first set of experiment for evaluating masked rationales
        fname = os.path.join(
            os.getcwd(),
            args["evaluation_dir"],
            args["thresholder"] + "-faithfulness-metrics.json"
        )


        if os.path.isfile(fname) == False:

            raise OSError(f"faithfulness metrics file does not exist at {fname} // rerun experiments in this file") from None
        
        ## retrieve predictions
        with open(fname, "r") as file : prediction_data = json.load(file)

        compute_faithfulness_(
            rationale_metadata=rationale_metadata,
            prediction_data=prediction_data
        )

        
        return