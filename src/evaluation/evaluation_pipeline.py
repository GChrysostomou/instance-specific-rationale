
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

from src.models.bert import BertClassifier_noise, BertClassifier_zeroout, bert, BertClassifier_attention
from src.variable_rationales.var_length_feat import get_rationale_metadata_
#from src.variable_rationales.var_type import select_between_types_
from src.evaluation.experiments.rationale_extractor import rationale_creator_, rationale_creator_interpolation_, extract_importance_, extract_lime_scores_, extract_shap_values_
from src.evaluation.experiments.erasure_tests import conduct_tests_2, conduct_tests_, conduct_experiments_noise_, conduct_experiments_zeroout_, conduct_experiments_attention_ #, conduct_experiments_attention_2
from src.evaluation.experiments.increasing_feature_scoring import compute_faithfulness_
import re


from src.evaluation.experiments.soft_erasure_tests import soft_conduct_tests_

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

    #def register_importance_(self, data, data_split_name, model = None):
    def register_importance_(self, data, data_split_name, tokenizer, max_seq_len, model = None): # debug by cass
        
        if model:


                        
            extract_importance_(
                    model = model, 
                    data_split_name = data_split_name,
                    data = data,
                    model_random_seed = self.model_random_seed
                )


            extract_lime_scores_(
                model = model, 
                data = data,
                data_split_name = data_split_name,
                model_random_seed = self.model_random_seed,
                no_of_labels = self.output_dims,
                max_seq_len = max_seq_len,
                tokenizer = tokenizer,
            )
            
            extract_shap_values_(
                model = model, 
                data = data,
                data_split_name = data_split_name,
                model_random_seed = self.model_random_seed,
                # no_of_labels = no_of_labels,
                # max_seq_len = max_seq_len,
                # tokenizer = tokenizer
            )

        else:


            for model_name in self.models:
                
                model = bert(
                    output_dim = self.output_dims
                )

                logging.info(f" *** loading model -> {model_name}")

                model.load_state_dict(torch.load(model_name, map_location=device))

                model.to(device)

                logging.info(f" *** succesfully loaded model -> {model_name}")

                self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

                extract_importance_(
                    model = model, 
                    data_split_name = data_split_name,
                    data = data,
                    model_random_seed = self.model_random_seed
                )

                extract_lime_scores_(
                    model = model, 
                    data = data,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    no_of_labels = data.nu_of_labels,
                    max_seq_len = data.max_len,
                    tokenizer = data.tokenizer,
                )

                extract_shap_values_(
                    model = model, 
                    data = data,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    #no_of_labels = data.nu_of_labels,
                    #max_seq_len = data.max_len,
                    #tokenizer = data.tokenizer
                )

        return

    def prepare_for_rationale_creation_(self,data):

        for i, model_name in enumerate(self.models):
            print('i = ', i)

            model = bert(
                output_dim = self.output_dims
            )


            logging.info(f" *** loading model -> {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model -> {model_name}")

            self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## train neglected as we are evaluating on dev and test
            for data_split_name, data_split in {"test":  data.test_loader # , # \
                                                
                                                # "dev":  data.dev_loader
                                                # added "train" only for creating FA --> "train": data.train_loader, \
                                                }.items():

                ## register importance scores if they do not exist
                self.register_importance_(
                    data = data_split,
                    data_split_name=data_split_name,
                    model = model,
                    #no_of_labels = data.nu_of_labels,
                    max_seq_len = data.max_len,
                    tokenizer = data.tokenizer  # comment out by cass
                )

                get_rationale_metadata_(
                    model = model, 
                    data_split_name = data_split_name,
                    data = data_split,
                    model_random_seed = self.model_random_seed
                )


                # select_between_types_(
                #     data_split_name = data_split_name,
                #     model_random_seed = self.model_random_seed
                # )
                # print(' DONE select between types')


        return

    def create_rationales_(self, data):

        
        for data_split_name, data_split in data.as_dataframes_().items():
            
            rationale_creator_(
                    data = data_split,
                    data_split_name = data_split_name,
                    tokenizer = data.tokenizer,
                    model_random_seed = self.model_random_seed,
                    )

        return


    def create_rationales_interpolation(self, data):

        
        for data_split_name, data_split in data.as_dataframes_().items():
            
            rationale_creator_interpolation_(
                    data = data_split,
                    data_split_name = data_split_name,
                    tokenizer = data.tokenizer,
                    model_random_seed = self.model_random_seed
                    )

        return


    def faithfulness_experiments_(self, data):
        
        for model_name in self.models:
            model = bert(
                output_dim = self.output_dims
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])
            
            ## check first if necessary data exists
            # fname = os.path.join(
            #     os.getcwd(),
            #     args["extracted_rationale_dir"],
            #     args["thresholder"],
            #     "test-rationale_metadata.npy"
            # )
            fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            "importance_scores",
            ""
        )
        
            fname = fname + f"test_importance_scores_{model_random_seed}.npy"


            if os.path.isfile(fname) == False:

                raise OSError(f"rationale metadata file does not exist at {fname} // rerun extract_rationales.py") from None
          



            ## train neglected as we are evaluating on dev and test
            for data_split_name, data_split in {"test":  data.test_loader,
                                                #"dev":  data.dev_loader,
                                                #"train":  data.train_loader
                                                }.items():

                conduct_tests_(
                    model = model, 
                    data = data_split,
                    model_random_seed = model_random_seed,
                    # split = data_split_name
                )

        return


    
    def faithfulness_experiments_interpolation_(self, data):
        
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
                output_dim = self.output_dims
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## train neglected as we are evaluating on dev and test
            for data_split_name, data_split in {"test":  data.test_loader
                                                #"dev":  data.dev_loader
                                                }.items():
            
                conduct_tests_(
                    model = model, 
                    data = data_split,
                    model_random_seed = model_random_seed,
                    # split = data_split_name
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
            args["thresholder"] + "-test-faithfulness-metrics.json"
        )


        if os.path.isfile(fname) == False:

            raise OSError(f"faithfulness metrics file does not exist at {fname} // rerun experiments in this file") from None
        
        ## retrieve predictions
        with open(fname, "r") as file : prediction_data = json.load(file)

        compute_faithfulness_(
            rationale_metadata=rationale_metadata,
            prediction_data=prediction_data,
            split_name = "test",
        )

        
        return


    def token_wise_performance(self):

        ## load rationale metadata for divergence scores
        ## WARNING // dependent on computing all the rationale metadata

        for data_split_name in {"test","dev"}:

            fname = os.path.join(
                os.getcwd(),
                args["extracted_rationale_dir"],
                args["thresholder"],
                f"{data_split_name}-rationale_metadata.npy"
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
                args["thresholder"] + f"-{data_split_name}-faithfulness-metrics.json"
            )


            if os.path.isfile(fname) == False:

                raise OSError(f"faithfulness metrics file does not exist at {fname} // rerun experiments in this file") from None
            
            ## retrieve predictions
            with open(fname, "r") as file : prediction_data = json.load(file)

            compute_faithfulness_(
                rationale_metadata=rationale_metadata,
                prediction_data=prediction_data,
                split_name=data_split_name
            )

        
        return


class evaluate_zeroout():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, model_path, output_dims, use_topk, normalise,
                ):
        
        """
        loads and holds a pretrained model
        """
        print(model_path, args["model_abbreviation"])
        self.models = glob.glob(model_path + args["model_abbreviation"] + "*.pt")
        self.output_dims = output_dims
        self.use_topk = use_topk
        self.normalise = normalise

        logging.info(f" *** there are {len(self.models)} models in :  {model_path}")

        if len(self.models) == 0:
            raise FileNotFoundError(
                f"*** no models in directory -> {model_path}")
    

    def faithfulness_experiments_(self, data):
        
        for model_name in self.models:
            
            model = BertClassifier_zeroout(
                output_dim = self.output_dims,
                # tasc = tasc_mech,
                # faithful_method='comp',
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))
            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")
            print('successfully loaded model for evaluating faithfulness')

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            conduct_experiments_zeroout_(
                model = model, 
                data = data.test_loader,
                model_random_seed = model_random_seed,
                #faithful_method = self.faithful_method,
                use_topk = self.use_topk,
                normalise= self.normalise,
            )

        return


class evaluate_noise():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, model_path, std, use_topk, output_dims, normalise,
                ):
        
        """
        loads and holds a pretrained model
        """
        print(model_path, args["model_abbreviation"])
        self.models = glob.glob(model_path + args["model_abbreviation"] + "*.pt")
        self.output_dims = output_dims
        #self.faithful_method = faithful_method
        #self.feature_name = feature_name
        self.std = std
        self.use_topk = use_topk
        self.normalise = normalise

        logging.info(f" *** there are {len(self.models)} models in :  {model_path}")

        if len(self.models) == 0:

            raise FileNotFoundError(
                f"*** no models in directory -> {model_path}"
            )
 
    # for soft in evaluate_soft
    def faithfulness_experiments_(self, data):
        
        for model_name in self.models:
            
            model = BertClassifier_noise(
                output_dim = self.output_dims,
                # faithful_method='comp',
                std = self.std,
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))
            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")
            print('successfully loaded model for evaluating faithfulness')

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            conduct_experiments_noise_(
                model = model, 
                data = data.test_loader,
                model_random_seed = model_random_seed,
                #faithful_method = self.faithful_method,
                std = self.std,
                use_topk = self.use_topk,
                normalise= self.normalise,
            )

        return


class evaluate_attention():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, model_path, output_dims, use_topk, normalise,
                # faithful_method = 'comp',
                # feature_name = 'attention',
                #std = 1
                ):
        
        """
        loads and holds a pretrained model
        """
        print(model_path, args["model_abbreviation"])
        self.models = glob.glob(model_path + args["model_abbreviation"] + "*.pt")
        self.output_dims = output_dims
        self.use_topk =use_topk
        self.normalise = normalise

        logging.info(f" *** there are {len(self.models)} models in :  {model_path}")

        if len(self.models) == 0:
            raise FileNotFoundError(
                f"*** no models in directory -> {model_path}")

    # for soft in evaluate_soft
    def faithfulness_experiments_(self, data):
        
        for model_name in self.models:
            
            model = BertClassifier_attention(
                output_dim = self.output_dims,
                #tasc = tasc_mech,
                # faithful_method='comp',
                #std = self.std,
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))
            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")
            print('successfully loaded model for evaluating faithfulness')

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            conduct_experiments_attention_(
                model = model, 
                data = data.test_loader,
                model_random_seed = model_random_seed,
                #faithful_method = self.faithful_method,
                #std = self.std,
                use_topk=self.use_topk,
                normalise= self.normalise,
            )

        return



class evaluate_interpolation():

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

    #def register_importance_(self, data, data_split_name, model = None):
    def register_importance_(self, data, data_split_name, no_of_labels, max_seq_len, tokenizer, model = None): # debug by cass
        
        if model:

            extract_importance_(
                    model = model, 
                    data_split_name = data_split_name,
                    data = data,
                    model_random_seed = self.model_random_seed
                )

            # extract_lime_scores_(
            #     model = model, 
            #     data = data,
            #     data_split_name = data_split_name,
            #     model_random_seed = self.model_random_seed,
            #     no_of_labels = no_of_labels,
            #     max_seq_len = max_seq_len,
            #     tokenizer = tokenizer,
            # )

            # extract_shap_values_(
            #     model = model, 
            #     data = data,
            #     data_split_name = data_split_name,
            #     model_random_seed = self.model_random_seed,
            #     # no_of_labels = no_of_labels,
            #     # max_seq_len = max_seq_len,
            #     # tokenizer = tokenizer
            # )

        else:

            for model_name in self.models:
                
                model = bert(output_dim = self.output_dims)

                logging.info(f" *** loading model -> {model_name}")

                model.load_state_dict(torch.load(model_name, map_location=device))

                model.to(device)

                logging.info(f" *** succesfully loaded model -> {model_name}")

                self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

                extract_importance_(
                    model = model, 
                    data_split_name = data_split_name,
                    data = data,
                    model_random_seed = self.model_random_seed
                )

                extract_lime_scores_(
                    model = model, 
                    data = data,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    no_of_labels = data.nu_of_labels,
                    max_seq_len = data.max_len,
                    tokenizer = data.tokenizer,
                )

                extract_shap_values_(
                    model = model, 
                    data = data,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    #no_of_labels = data.nu_of_labels,
                    #max_seq_len = data.max_len,
                    #tokenizer = data.tokenizer
                )

        return

    def prepare_for_rationale_creation_(self,data):

        for i, model_name in enumerate(self.models):
            print('i = ', i)

            model = bert(
                output_dim = self.output_dims
            )


            logging.info(f" *** loading model -> {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model -> {model_name}")

            self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## train neglected as we are evaluating on dev and test
            get_rationale_metadata_(model = model, data_split_name = "test", data = data.test_loader, model_random_seed = self.model_random_seed)
        return

    def create_rationales_(self, data):
        
        for data_split_name, data_split in data.as_dataframes_().items():
            
            try:

                rationale_creator_(
                    data = data_split,
                    data_split_name = data_split_name,
                    tokenizer = data.tokenizer,
                    #variable = False
                )

            except:

                raise NotImplementedError



        return



    def faithfulness_experiments_(self, data):
        
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
                output_dim = self.output_dims
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## train neglected as we are evaluating on dev and test
            for data_split_name, data_split in {"test":  data.test_loader
                                                #"dev":  data.dev_loader
                                                }.items():
            
                conduct_tests_(
                    model = model, 
                    data = data_split,
                    model_random_seed = model_random_seed,
                    # split = data_split_name
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
            args["thresholder"] + "-test-faithfulness-metrics.json"
        )


        if os.path.isfile(fname) == False:

            raise OSError(f"faithfulness metrics file does not exist at {fname} // rerun experiments in this file") from None
        
        ## retrieve predictions
        with open(fname, "r") as file : prediction_data = json.load(file)

        compute_faithfulness_(
            rationale_metadata=rationale_metadata,
            prediction_data=prediction_data,
            split_name = "test",
        )

        
        return



    def prepare_for_rationale_creation_for_interpolation(self,data):

        for i, model_name in enumerate(self.models):
            print('i = ', i)

            model = bert(
                output_dim = self.output_dims
            )


            logging.info(f" *** loading model -> {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model -> {model_name}")

            self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## train neglected as we are evaluating on dev and test
            get_rationale_metadata_(model = model, data_split_name = "test", data = data.test_loader, model_random_seed = self.model_random_seed)
        return

    def create_rationales_for_interpolation(self, data, topk):
        

        for data_split_name, data_split in data.as_dataframes_().items():
            
            rationale_creator_interpolation_(
                    data = data_split,
                    data_split_name = data_split_name,
                    tokenizer = data.tokenizer,
                    topk=topk,
                )



        return

