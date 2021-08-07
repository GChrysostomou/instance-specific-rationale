import torch
from torch import nn
import json
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import config.cfg
from config.cfg import AttrDict
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nn.deterministic = True
torch.backends.cudnn.benchmark = False
    

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)

from src.evaluation import thresholders
from src.common_code.useful_functions import wpiece2word 


def extract_importance_(model, data, data_split_name, model_random_seed):

    """
        Info: computes the average fraction of tokens required to cause a decision flip (prediction change)
        Input:
            model : pretrained model
            data : torch.DataLoader loaded data
            save_path : path to save the results
        Output:
            saves the results to a csv file under the save_path
    """

    desc = f'registering importance scores for {data_split_name} -> id'
    

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)

    scorenames = fname + data_split_name + f"_importance_scores_{model_random_seed}.npy"

    # check if importance scores exist first to avoid unecessary calculations
    if os.path.exists(scorenames):

        print(f"importance scores already saved in -> {scorenames}")

        return
    

    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    
    feature_attribution = {}

    for batch in data:
        
        model.eval()
        model.zero_grad()

        batch = {
                "annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "query_mask" : batch["query_mask"].squeeze(1).to(device),
                "retain_gradient" : True
            }
            
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
        yhat, attentions =  model(**batch)

        yhat.max(-1)[0].sum().backward(retain_graph = True)

        #embedding gradients
        embed_grad = model.wrapper.model.embeddings.word_embeddings.weight.grad
        g = embed_grad[batch["input_ids"].long()]


        em = model.wrapper.model.embeddings.word_embeddings.weight[batch["input_ids"].long()]

        gradients = (g* em).sum(-1).abs() * batch["query_mask"].float()

        integrated_grads = model.integrated_grads(
                original_grad = g, 
                original_pred = yhat.max(-1),
                **batch    
        )

        normalised_random = torch.randn(attentions.shape).to(device)

        normalised_random = torch.masked_fill(normalised_random, ~batch["query_mask"].bool(), float("-inf"))

        # normalised integrated gradients of input
        normalised_ig = torch.masked_fill(integrated_grads, ~batch["query_mask"].bool(), float("-inf"))

        # normalised gradients of input
        normalised_grads = torch.masked_fill(gradients, ~batch["query_mask"].bool(), float("-inf"))

        # normalised attention
        normalised_attentions = torch.masked_fill(attentions, ~batch["query_mask"].bool(), float("-inf"))

        # retrieving attention*attention_grad
        attention_gradients = model.weights_or.grad[:,:,0,:].mean(1)
        
        attention_gradients =  (attentions * attention_gradients)

        # softmaxing due to negative attention gradients 
        # therefore we receive also negative values and as such
        # the pad and unwanted tokens need to be converted to -inf 
        normalised_attention_grads = torch.masked_fill(attention_gradients, ~batch["query_mask"].bool(), float("-inf"))

        for _i_ in range(attentions.size(0)):

            annotation_id = batch["annotation_id"][_i_]
            ## storing feature attributions
            feature_attribution[annotation_id] = {
                "random" : normalised_random[_i_].cpu().detach().numpy(),
                "attention" : normalised_attentions[_i_].cpu().detach().numpy(),
                "gradients" : normalised_grads[_i_].cpu().detach().numpy(),
                "ig" : normalised_ig[_i_].cpu().detach().numpy(),
                "scaled attention" : normalised_attention_grads[_i_].cpu().detach().numpy()
            }

        pbar.update(data.batch_size)

    ## save them
    np.save(scorenames, feature_attribution)

    print(f"model dependent importance scores stored in -> {scorenames}")

    return

from src.evaluation.experiments.lime_predictor import predictor
from lime.lime_text import LimeTextExplainer

def extract_lime_scores_(model, data, data_split_name, model_random_seed, no_of_labels, max_seq_len, tokenizer):

    
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    fname += f"{data_split_name}_importance_scores_{model_random_seed}.npy"

    ## retrieve importance scores
    importance_scores = np.load(fname, allow_pickle = True).item()

    lime_predictor = predictor(
        model = model, 
        tokenizer = tokenizer, 
        seq_length = max_seq_len
    )
    
    explainer = LimeTextExplainer(class_names=list(range(no_of_labels)), split_expression=" ")

    train_ls = {}
    ## we are interested in token level features
    for batch in data:
        
        for _j_ in range(batch["input_ids"].size(0)):

            input_ids = batch["input_ids"][_j_].squeeze(0)
            annotation_id = batch["annotation_id"][_j_]

            if args.query:
                
                length = (batch["attention_mask"][_j_] != 0).sum().detach().cpu().item()

            else:

                length = batch["lengths"][_j_].detach().cpu().item()

            train_ls[annotation_id] = {
                "example" : " ".join(tokenizer.convert_ids_to_tokens(input_ids)),
                "split example" : " ".join(tokenizer.convert_ids_to_tokens(input_ids)[:length]),
                "query mask" : batch["query_mask"][_j_].squeeze(0).detach().cpu().numpy(),
                "annotation_id" : annotation_id,
                "length" : length
            }

    desc =  f"computing lime scores for -> {data_split_name}"
    pbar = trange(len(train_ls.keys()), desc=desc, leave=True)

    for annot_id in train_ls.keys():

        ## skip to save time if we already run lime (VERY EXPENSIVE)
        if "lime" in importance_scores[annot_id]:

            continue

        exp = explainer.explain_instance(
            train_ls[annot_id]["split example"], 
            lime_predictor.predictor, 
            num_samples = 500, 
            num_features = len(set(train_ls[annot_id]["split example"])) 
        )

        words = dict(exp.as_list())

        feature_importance = np.asarray([words[x] if x in words else 0. for x in train_ls[annot_id]["example"].split()])

        feature_importance = np.ma.array(
            feature_importance.tolist(), 
            mask = (train_ls[annot_id]["query mask"] == 0).astype(np.long).tolist(), 
            fill_value = float("-inf")
        )

        pbar.update(1)

        importance_scores[annot_id]["lime"] = feature_importance.filled()


     ## save them
    np.save(fname, importance_scores)

    print(f"appended lime scores in -> {fname}")

    return


from src.evaluation.experiments.shap_predictor import ShapleyModelWrapper
from captum.attr import DeepLiftShap

def extract_shap_values_(model, data, data_split_name, model_random_seed, no_of_labels, max_seq_len, tokenizer):
    
    
    fname = os.path.join(
        os.getcwd(),
        args["data_dir"],
        "importance_scores",
        ""
    )

    fname += f"{data_split_name}_importance_scores_{model_random_seed}.npy"

    ## retrieve importance scores
    importance_scores = np.load(fname, allow_pickle = True).item()

    explainer = DeepLiftShap(ShapleyModelWrapper(model))

    pbar = trange(len(data) * data.batch_size, desc=f"extracting shap scores for -> {data_split_name}", leave=True)

    ## we are interested in token level features
    for batch in data:

        model.eval()
        model.zero_grad()

        batch = {
            "annotation_id" : batch["annotation_id"],
            "input_ids" : batch["input_ids"].squeeze(1).to(device),
            "lengths" : batch["lengths"].to(device),
            "labels" : batch["label"].to(device),
            "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
            "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
            "query_mask" : batch["query_mask"].squeeze(1).to(device),
            "special_tokens" : batch["special tokens"],
            "retain_gradient" : False ## we do not need it
        }
            
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
        
        original_prediction, _ =  model(**batch)

        embeddings = model.wrapper.model.embeddings.word_embeddings.weight[batch["input_ids"].long()]
        baseline = torch.zeros_like(embeddings, requires_grad = True).to(device)

        attribution = explainer.attribute(embeddings.requires_grad_(True), baseline, target = original_prediction.argmax(-1))

        attribution = attribution.sum(-1)

        attribution = torch.masked_fill(
            attribution, 
            (batch["query_mask"] == 0).bool(), 
            float("-inf")
        )

        for _i_ in range(attribution.size(0)):

            annotation_id = batch["annotation_id"][_i_]

            importance_scores[annotation_id]["shap"] = attribution[_i_].detach().cpu().numpy()


        pbar.update(data.batch_size)

     ## save them
    np.save(fname, importance_scores)

    print(f"appended shap scores in -> {fname}")

    return

def rationale_creator_(data, data_split_name, variable, tokenizer):

    ## get the thresholder fun
    thresholder = getattr(thresholders, args["thresholder"])

    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        ""
    )

    fname += data_split_name + "-rationale_metadata.npy"

    ## retrieve importance scores
    rationale_metadata = np.load(fname, allow_pickle = True).item()

    ## filter only relevant parts in our dataset
    data = data[["text", "annotation_id", "exp_split", "label", "label_id"]]

    data["tokenized_text"] = data["text"]

    annotation_text = dict(data[["annotation_id", "tokenized_text"]].values)

    del data["tokenized_text"]

    ## time to register rationales
    for feature_attribution in {"attention", "gradients", "ig", "scaled attention"}:
        

        temp_registry = {}

        for annotation_id, sequence_text in annotation_text.items():

            ## check if there is any padding which could affect our process and remove
            seq_length = (np.asarray(sequence_text) != 0).sum()

            sequence_importance = rationale_metadata[annotation_id][feature_attribution]["importance scores"][:seq_length]
            sequence_text = sequence_text[:len(sequence_importance)]

            if variable:

                # untokenize sequence and sequence importance scores
                sequence_text, sequence_importance = wpiece2word(
                    tokenizer = tokenizer, 
                    sentence = sequence_text, 
                    weights = sequence_importance
                )

                rationale_indxs = thresholder(
                    scores = sequence_importance, 
                    original_length = len(sequence_text) - 2,
                    rationale_length =  rationale_metadata[annotation_id][feature_attribution]["variable rationale ratio"]
                )

            else:

                ## untokenize sequence and sequence importance scores
                sequence_text, sequence_importance = wpiece2word(
                    tokenizer = tokenizer, 
                    sentence = sequence_text, 
                    weights = sequence_importance
                )

                rationale_indxs = thresholder(
                    scores = sequence_importance, 
                    original_length = len(sequence_text) -2,
                    rationale_length = args["rationale_length"]
                )

            rationale = sequence_text[rationale_indxs]

            temp_registry[annotation_id] = " ".join(rationale)

        data.text = data.annotation_id.apply(
            lambda x : temp_registry[x]
        )

        fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            args["thresholder"],
            "data",
            ""
        )

        os.makedirs(fname, exist_ok=True)

        
        if variable:
            fname += "var_len-" + feature_attribution + "-" + data_split_name + ".csv"
        else:
            fname += feature_attribution + "-" + data_split_name + ".csv"

        print(f"saved in -> {fname}")

        data.to_csv(fname)

    ## now to save our (fixed-len + var-feat) and (var-len + var-feat rationales)

    temp_registry = {}

    if variable: for_our_approach = "var-len_var-feat"
    else: for_our_approach = "fixed-len_var-feat"

    for annotation_id, sequence_text in annotation_text.items():
        

        ## check if there is any padding which could affect our process and remove
        seq_length = (np.asarray(sequence_text) != 0).sum()

        sequence_importance = rationale_metadata[annotation_id][for_our_approach]["importance scores"][:seq_length]
        sequence_text = sequence_text[:len(sequence_importance)]

        # untokenize sequence and sequence importance scores
        sequence_text, sequence_importance = wpiece2word(
            tokenizer = tokenizer, 
            sentence = sequence_text, 
            weights = sequence_importance
        )

        rationale_indxs = thresholder(
            scores = sequence_importance, 
            original_length = len(sequence_text) - 2,
            rationale_length =  rationale_metadata[annotation_id][for_our_approach]["variable rationale ratio"]
        )

        rationale = sequence_text[rationale_indxs]

        temp_registry[annotation_id] = " ".join(rationale)

    data.text = data.annotation_id.apply(
        lambda x : temp_registry[x]
    )

    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        args["thresholder"],
        "data",
        ""
    )

    os.makedirs(fname, exist_ok=True)

    
    if variable:
        fname += "var-len_var-feat-" + data_split_name + ".csv"
    else:
        fname += "fixed-len_var-feat-" + "-" + data_split_name + ".csv"

    print(f"saved in -> {fname}")

    data.to_csv(fname)


    return


