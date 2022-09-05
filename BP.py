import logging

logging.basicConfig(level=logging.INFO)
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import argparse
import numpy as np
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    NormalFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.scenario.scenario import Scenario

# __copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
# __license__ = "3-clause BSD"
import datetime
import os, sys


date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])


digits = load_digits()



parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "evinf", 
    #choices = ["sst", "evinf", "agnews", "multirc", "evinf_FA"]
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
    "--seed",   
    type = int, 
    help = "random seed for experiment"
)

parser.add_argument(
    '--evaluate_models', 
    help='test predictive performance in and out of domain', 
    action='store_true',
    default=False,
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

### used only for data stats
data_dir_plain = user_args["data_dir"]


log_dir = "experiment_logs/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" +  date_time + "/"
config_dir = "experiment_config/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" + date_time + "/"


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

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")



from src.data_functions.dataholder import classification_dataholder as dataholder
from src.tRpipeline import test_predictive_performance_TL, train_and_save, test_predictive_performance, keep_best_model_
from src.data_functions.useful_functions import describe_data_stats

from sklearn.model_selection import train_test_split
import pandas as pd
import json


feature = 'ig'
len = ''
thresh = 'topk' # ["topk", "contigious"]
dataset = str(args["dataset"])


topk = f"faithfulness_metrics/{dataset}/{thresh}-test-faithfulness-metrics.json"
conti = f"faithfulness_metrics/{dataset}/{thresh}-test-faithfulness-metrics.json"
with open(topk) as topk_f:
    topk_dict = json.load(topk_f)

with open(topk) as conti_f:
    conti_dict = json.load(conti_f)


print(data.columns) # Index(['Unnamed: 0', 'annotation_id', 'exp_split', 'label', 'label_id',
       # 'document', 'query', 'full text doc', 'full text prediction',
       # 'rationale length'],
quit()


# Target Algorithm
def mlp_from_cfg(cfg, seed):
    """
    Creates a MLP classifier from sklearn and fits the given data on it.

    Parameters
    ----------
    cfg: Configuration
        configuration chosen by smac
    seed: int or RandomState
        used to initialize the rf's random generator
    budget: float
        used to set max iterations for the MLP

    Returns
    -------
    float
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # mlp = MLPClassifier(
        #     hidden_layer_sizes=[cfg["n_neurons"]] * cfg["n_layer"],
        #     solver=cfg["optimizer"],
        #     batch_size=cfg["batch_size"],
        #     activation=cfg["activation"],
        #     learning_rate_init=cfg["learning_rate_init"],
        #     random_state=seed,
        # )

        # # returns the cross validation accuracy
        # cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
        # score = cross_val_score(mlp, digits.data, digits.target, cv=cv, error_score="raise")
        
        feature = cfg['feature']
        len = cfg['len']
        thresh = cfg['thresholder'] # ["topk", "contigious"]


        fname = f"{extracted_rationale_dir}/{task_name}/{thresh}/test-rationale_metadata.npy"
        data = np.load(fname, allow_pickle = True).item()
        print(data)

    return 1 - np.mean(score)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    # We do not have an educated belief on the number of layers beforehand
    # As such, the prior on the HP is uniform
    #n_layer = UniformIntegerHyperparameter("n_layer", lower=1, upper=5)

    # We believe the optimal network is likely going to be relatively wide,
    # And place a Beta Prior skewed towards wider networks in log space
    #n_neurons = BetaIntegerHyperparameter("n_neurons", lower=8, upper=1024, alpha=4, beta=2, log=True)

    # We believe that ReLU is likely going to be the optimal activation function about
    # 60% of the time, and thus place weight on that accordingly
    feature = CategoricalHyperparameter(
        "feature", ["deeplift", "lime", "attention", "ig", "gradients","scaled attention", "random"], 
        # weights=[1, 1, 3], 
        default_value="scaled attention",
    )

    # Moreover, we believe ADAM is the most likely optimizer
    thresh = CategoricalHyperparameter("thresh", ["topk", "contigious"], 
    #weights=[1, 2], 
    default_value="topk",
    )

    # We do not have an educated opinion on the batch size, and thus leave it as-is
    #batch_size = UniformIntegerHyperparameter("batch_size", 16, 512, default_value=128)

    # We place a log-normal prior on the learning rate, so that it is centered on 10^-3,
    # with one unit of standard deviation per multiple of 10 (in log space)
    #learning_rate_init = NormalFloatHyperparameter(
    #     "learning_rate_init", lower=1e-5, upper=1.0, mu=np.log(1e-3), sigma=np.log(10), log=True
    # )

    # Add all hyperparameters at once:
    cs.add_hyperparameters([feature, thresh])

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "runcount-limit": 20,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "deterministic": "true",
            "limit_resources": True,  # Uses pynisher to limit memory and runtime
            # Alternatively, you can also disable this.
            # Then you should handle runtime and memory yourself in the TA
            "cutoff": 30,  # runtime limit for target algorithm
            "memory_limit": 3072,  # adapt this to reasonable value for your hardware
        }
    )

    # The rate at which SMAC forgets the prior.
    # The higher the value, the more the prior is considered.
    # Defaults to # n_iterations / 10
    user_prior_kwargs = {"decay_beta": 1.5}

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=mlp_from_cfg,
        # This flag is required to conduct the optimisation using priors over the optimum
        user_priors=True,
        user_prior_kwargs=user_prior_kwargs,
        # Using random configurations will cause the initialization to be samples drawn from the prior
        initial_design=RandomConfigurations,
    )

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(), seed=0)[1]

    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, seed=0)[1]

    print("Optimized Value: %.4f" % inc_value)