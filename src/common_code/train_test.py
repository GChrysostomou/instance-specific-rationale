from tqdm import trange
import torch
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
import config.cfg
from config.cfg import AttrDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1

import json 
import logging
with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


class checkpoint_holder(object):

    """
    holds checkpoint information for the 
    training of models
    """

    def __init__(self, save_model_location : str):

        self.dev_loss = float("inf")
        self.save_model_location = save_model_location
        self.point = 0
        self.storer = {}

    def _store(self, model, point : int, epoch : int, dev_loss, dev_results : dict) -> dict:
        if self.dev_loss > dev_loss:
            
            self.dev_loss = dev_loss
            self.point = point
            self.storer = dev_results
            self.storer["epoch"] = epoch + 1
            self.storer["point"] = self.point
            self.storer["dev_loss"] = self.dev_loss

            torch.save(model.state_dict(), self.save_model_location)

        return self.storer
        
import config.cfg
from config.cfg import AttrDict



def test_model(model, loss_function, data, save_output_probs = True, random_seed = None, for_rationale = False, variable = False):
    
    """ 
    Model predictive performance on unseen data
    Input: 
        "model" : initialised pytorch model
        "loss_function" : loss function to calculate loss at output
        "data" : unseen data (test)
    Output:
        "results" : classification results
        "loss" : normalised loss on test data
    """

    predicted = [] 
    
    actual = []
    
    total_loss = 0

    if save_output_probs:

        to_save_probs = {}
  
    with torch.no_grad():

        model.eval()
    
        for batch in data:
            
            batch = {
                "annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "retain_gradient" : False
            }
            
            assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
            
            yhat, _ =  model(**batch)  # get results

            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)

            if save_output_probs:

                for _j_ in range(yhat.size(0)):

                    to_save_probs[batch["annotation_id"][_j_]] = {}
                    to_save_probs[batch["annotation_id"][_j_]]["predicted"] = yhat[_j_].detach().cpu().numpy()
                    to_save_probs[batch["annotation_id"][_j_]]["actual"] = batch["labels"][_j_].detach().cpu().item()

                    if args.inherently_faithful:

                        leng = batch["lengths"][_j_]

                        to_save_probs[batch["annotation_id"][_j_]]["rationale"] = model.sample_z[_j_][:leng].detach().cpu().numpy()
                        to_save_probs[batch["annotation_id"][_j_]]["full text length"] = leng.detach().cpu().item()

            if args.inherently_faithful:

                loss = model._joint_rationale_objective(
                    predicted_logits = yhat,
                    actual_labels = batch["labels"]
                )

                loss = torch.abs(loss)

            else:

                loss = loss_function(yhat, batch["labels"]) 

            total_loss += loss.item()
            
            _, ind = torch.max(yhat, dim = 1)
    
            predicted.extend(ind.cpu().numpy())
    
            actual.extend(batch["labels"].cpu().numpy())

        results = classification_report(actual, predicted, output_dict = True)

    ### random seed just used for saving probabilities
    if save_output_probs:

        if for_rationale:

            if variable:

                fname = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] +"-" \
                    + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +"-variable.npy"

            else:

                fname = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] +"-" \
                    + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +".npy"

        else:

            if variable:

                fname = args["model_dir"] + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +"-variable.npy"

            else:

                fname = args["model_dir"] + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +".npy"


        np.save(fname, to_save_probs)

        return results, (total_loss * data.batch_size / len(data)) , to_save_probs

    return results, (total_loss * data.batch_size / len(data)) 


def train_model(model, training, development, loss_function, optimiser, seed,
            run, epochs = 10, cutoff_len = 2, cutoff = True, save_folder  = None, 
            ):
    
    """ 
    Trains the model and saves it at required path
    Input: 
        "model" : initialised pytorch model
        "training" : training dataset ---> dataloader
        "development" : development dataset
        "loss_function" : loss function to calculate loss at output
        "optimiser" : pytorch optimiser (Adam)
        "run" : which of the 5 training runs is this?
        "epochs" : number of epochs to train the model
        "cutoff" : early stopping (default False)
        "cutoff_len" : after how many increases in devel loss to cut training
        "save_folder" : folder to save checkpoints
    Output:
        "saved_model_results" : results for best checkpoint of this run
        "results_for_run" : analytic results for all epochs during this run
    """

    results = []
    
    results_for_run = ""
    
    pbar = trange(len(training) *epochs, desc='running for seed ' + run, leave=True, 
    bar_format = "{l_bar}{bar}{elapsed}<{remaining}")
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    checkpoint = checkpoint_holder(save_model_location = save_folder)

    total_steps = len(training) * args["epochs"]
    scheduler = get_linear_schedule_with_warmup(
                                                optimiser,
                                                num_warmup_steps=int(len(training)*.1),
                                                num_training_steps=total_steps
                                                )  
    every = round(len(training) / 3)

    logging.info("***************************************")
    logging.info("Training on seed {}".format(run))
    logging.info("*saving checkpoint every {} iterations".format(every))

    if every == 0: every = 1

    model.train()

    for epoch in range(epochs):
        
        total_loss = 0

        checks = 0

               
        for batch in training:
          
            
            model.zero_grad()


            batch = {
                "annotation_id" : batch["annotation_id"],  # add from SoftComp, by Cass
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "retain_gradient" : False,
            }

            assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
            
            yhat, _ =  model(**batch) # return logits, self.weights

            # print(f"==>> yhat.shape: {yhat.shape}", yhat)
            # print("==>> yhat.shape:", batch["labels"].size(), batch["labels"])


            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)
            
            if args.inherently_faithful:

                loss = model._joint_rationale_objective(
                    predicted_logits = yhat,
                    actual_labels = batch["labels"]
                )

            else:

                # print("==>> batch[labels]",batch["labels"])
                # print(f"==>> yhat: {yhat.shape}, {yhat}")
                # yhat = yhat[0]
                loss = loss_function(yhat, batch["labels"]) 



            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.)
            
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            pbar.update(1)
            pbar.refresh()
                
            if checks % every == 0:
                # def test_model(model, loss_function, data, save_output_probs = True, random_seed = None, for_rationale = False, variable = False):
                dev_results, dev_loss, dev_prob = test_model(
                    model = model, 
                    loss_function = loss_function, 
                    data = development,
                )

                checkpoint_results = checkpoint._store(
                    model = model, 
                    point = checks, 
                    epoch = epoch, 
                    dev_loss = dev_loss, 
                    dev_results = dev_results)


            checks += 1

        dev_results, dev_loss, dev_prob = test_model(
                                                model, 
                                                loss_function, 
                                                development
                                            )    

        results.append([epoch, dev_results["macro avg"]["f1-score"], dev_loss, dev_results])
        
        logging.info("*** epoch - {} | train loss - {} | dev f1 - {} | dev loss - {}".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2)))

        
        results_for_run += "epoch - {} | train loss - {} | dev f1 - {} | dev loss - {} \n".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2))

    return checkpoint_results, results_for_run


def train_model_mt5(model, training, development, loss_function, optimiser, seed,
            run, epochs = 10, cutoff_len = 2, cutoff = True, save_folder  = None, 
            ):
    
    """ 
    Trains the model and saves it at required path
    Input: 
        "model" : initialised pytorch model
        "training" : training dataset ---> dataloader
        "development" : development dataset
        "loss_function" : loss function to calculate loss at output
        "optimiser" : pytorch optimiser (Adam)
        "run" : which of the 5 training runs is this?
        "epochs" : number of epochs to train the model
        "cutoff" : early stopping (default False)
        "cutoff_len" : after how many increases in devel loss to cut training
        "save_folder" : folder to save checkpoints
    Output:
        "saved_model_results" : results for best checkpoint of this run
        "results_for_run" : analytic results for all epochs during this run
    """

    results = []
    
    results_for_run = ""
    
    pbar = trange(len(training) *epochs, desc='running for seed ' + run, leave=True, 
    bar_format = "{l_bar}{bar}{elapsed}<{remaining}")
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    checkpoint = checkpoint_holder(save_model_location = save_folder)

    total_steps = len(training) * args["epochs"]
    scheduler = get_linear_schedule_with_warmup(
                                                optimiser,
                                                num_warmup_steps=int(len(training)*.1),
                                                num_training_steps=total_steps
                                                )  
    every = round(len(training) / 3)

    logging.info("***************************************")
    logging.info("Training on seed {}".format(run))
    logging.info("*saving checkpoint every {} iterations".format(every))

    if every == 0: every = 1

    model.train()

    for epoch in range(epochs):
        
        total_loss = 0

        checks = 0

               
        for i, batch in enumerate(training):
            # if i ==0: 
            #     print(f' batch example, the len of batch is {len(batch)} the first data in the first batch')
            #     print(batch.keys())
            
            
            model.zero_grad()


            batch = {
                "annotation_id" : batch["annotation_id"],  # add from SoftComp, by Cass

                ########## into model forward arguments 1
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "labels": batch["label"].to(device),

                "output_attentions": True,
                "output_hidden_states": True,
                "return_dict": True,

                "decoder_input_ids" : batch["decoder_input_ids"].squeeze(1).to(device),
                "decoder_attention_mask" : batch["decoder_attention_mask"].squeeze(1).to(device),

                # "head_mask" #: Optional[torch.FloatTensor] = None,
                # "decoder_head_mask" #: Optional[torch.FloatTensor] = None,
                # "cross_attn_head_mask #": Optional[torch.Tensor] = None,
                # "encoder_outputs"#: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                # "past_key_values"#: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                # "inputs_embeds"#: Optional[torch.Tensor] = None,
                # "decoder_inputs_embeds"#: Optional[torch.Tensor] = None,
                # "use_cache"#: Optional[bool] = None,
                # "output_attentions"#: Optional[bool] = None,
                # "output_hidden_states"#: Optional[bool] = None,
                # "return_dict"#: Optional[bool] = None,

                ##################

                "lengths" : batch["lengths"].to(device),
                "retain_gradient" : True, # return seq2seq output
                "label_input_indexd_dict": batch["label_input_indexd_dict"] 
                # label_input_indexd_dict[id_in_num_format] = label_input_id_index
            }

            assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
            
            outputs =  model(**batch) # return logits, self.weights
            # print(f"==>> outputs 0 shape: {outputs[0].shape}")
            # print(f"==>> outputs 1 : {outputs[1].shape}")
            # chuan jin lai de model is a mt5, mt5 includes mt5wrapper, 
            # mt5         returns logits, self.weights
            # mt5wrapper  returns decoder_outputs + encoder_outputs if return_dict=True

            yhat = outputs[0]
            #print(f"==>> TRAINING yhat.shape: {yhat.shape}")


            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)
            
            if args.inherently_faithful:

                loss = model._joint_rationale_objective(
                    predicted_logits = yhat,
                    actual_labels = batch["labels"]
                )

            else:
                loss = loss_function(yhat, batch["labels"]) 


            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.)
            
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            pbar.update(1)
            pbar.refresh()
            
                
            if checks % every == 0:
                # def test_model(model, loss_function, data, save_output_probs = True, random_seed = None, for_rationale = False, variable = False):
                if args['model_abbreviation'] == 't5m': 
                    dev_results, dev_loss, dev_prob = test_model_mt5(
                    model = model, 
                    loss_function = loss_function, 
                    data = development,
                )
                else:
                    dev_results, dev_loss, dev_prob = test_model(
                    model = model, 
                    loss_function = loss_function, 
                    data = development,
                )

                checkpoint_results = checkpoint._store(
                    model = model, 
                    point = checks, 
                    epoch = epoch, 
                    dev_loss = dev_loss, 
                    dev_results = dev_results)


            checks += 1

        dev_results, dev_loss, dev_prob = test_model_mt5(
                                                model, 
                                                loss_function, 
                                                development
                                            )    

        results.append([epoch, dev_results["macro avg"]["f1-score"], dev_loss, dev_results])
        
        logging.info("*** epoch - {} | train loss - {} | dev f1 - {} | dev loss - {}".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2)))

        
        results_for_run += "epoch - {} | train loss - {} | dev f1 - {} | dev loss - {} \n".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2))

    return checkpoint_results, results_for_run




def test_model_mt5(model, loss_function, data, save_output_probs = True, random_seed = None, for_rationale = False, variable = False):
    
    """ 
    Model predictive performance on unseen data
    Input: 
        "model" : initialised pytorch model
        "loss_function" : loss function to calculate loss at output
        "data" : unseen data (test)
    Output:
        "results" : classification results
        "loss" : normalised loss on test data
    """

    predicted = [] 
    
    actual = []
    
    total_loss = 0

    if save_output_probs:

        to_save_probs = {}
  
    with torch.no_grad():

        model.eval()
    
        for batch in data:
            
            batch = {
                "annotation_id" : batch["annotation_id"], 

                "input_ids" : batch["input_ids"].squeeze(1).to(device),  
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "labels" : batch["label"].to(device),

                "output_attentions": True,
                "output_hidden_states": True,
                "return_dict": True,

                "decoder_input_ids" :torch.zeros(batch["decoder_input_ids"].size(), dtype=int).squeeze(1).to(device),
                "decoder_attention_mask" : batch["decoder_attention_mask"].squeeze(1).to(device),

                "lengths" : batch["lengths"].to(device),
                "retain_gradient" : False, # return seq2seq output
                "label_input_indexd_dict": batch["label_input_indexd_dict"]
            }
            
            assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
            
            outputs =  model(**batch)
            # print(f"==>> outputs[0].shape: {outputs[0].shape}")
            # print(f"==>> outputs[1].shape: {outputs[1].shape}")
            # print(f"==>> TESTING outputs: {outputs}")
            
            yhat = outputs[0]
            # print("".center(50, "-"))
            # print("".center(50, "-"))
            # print("".center(50, "-"))
            # print(f"==>> TESTING yhat: {yhat}")

            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)

            if save_output_probs:

                for _j_ in range(yhat.size(0)):

                    to_save_probs[batch["annotation_id"][_j_]] = {}
                    to_save_probs[batch["annotation_id"][_j_]]["predicted"] = yhat[_j_].detach().cpu().numpy()
                    to_save_probs[batch["annotation_id"][_j_]]["actual"] = batch["labels"][_j_].detach().cpu().item()

                    if args.inherently_faithful:

                        leng = batch["lengths"][_j_]

                        to_save_probs[batch["annotation_id"][_j_]]["rationale"] = model.sample_z[_j_][:leng].detach().cpu().numpy()
                        to_save_probs[batch["annotation_id"][_j_]]["full text length"] = leng.detach().cpu().item()

            if args.inherently_faithful:

                loss = model._joint_rationale_objective(
                    predicted_logits = yhat,
                    actual_labels = batch["labels"]
                )

                loss = torch.abs(loss)

            else:

                loss = loss_function(yhat, batch["labels"]) 
            # print("".center(50, "-"))
            # print("".center(50, "-"))
            # print("".center(50, "-"))
            
            
            # print(batch["labels"])
            # print(yhat)
            total_loss += loss.item()
            
            
            _, ind = torch.max(yhat, dim = 1)
    
            predicted.extend(ind.cpu().numpy())
    
            actual.extend(batch["labels"].cpu().numpy())

        results = classification_report(actual, predicted, output_dict = True)

    ### random seed just used for saving probabilities
    if save_output_probs:

        if for_rationale:

            if variable:

                fname = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] +"-" \
                    + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +"-variable.npy"

            else:

                fname = os.path.join(args["model_dir"], args["thresholder"], "") + args["importance_metric"] +"-" \
                    + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +".npy"

        else:

            if variable:

                fname = args["model_dir"] + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +"-variable.npy"

            else:

                fname = args["model_dir"] + args["model_abbreviation"] + "-output_seed-" + str(random_seed) +".npy"


        np.save(fname, to_save_probs)

        return results, (total_loss * data.batch_size / len(data)) , to_save_probs

    return results, (total_loss * data.batch_size / len(data)) 

