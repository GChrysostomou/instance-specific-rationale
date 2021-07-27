import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig
import json 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
            args = AttrDict(json.load(f))


class generator(nn.Module):
    def __init__(self, dropout=0.1):
        
        super(generator, self).__init__()

        """
        GENERATOR FOR LEI ET AL MODEL
        """

        self.dropout = dropout

        self.gen_config = AutoConfig.from_pretrained(
            args["model"], 
            output_attentions = True
        )   
        
        self.bert_generator = AutoModel.from_pretrained(
                args["model"], 
                config=self.gen_config
            )
        

        self.dropout = nn.Dropout(p = self.dropout)

        self.z_layer = nn.Linear(self.bert_generator.config.hidden_size, 1, bias = False)
        torch.nn.init.xavier_uniform_(self.z_layer.weight)

    def forward(self, **inputs):

        self.output, pooled_output, attention_weights = self.bert_generator(
            inputs["sentences"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"]
        )


        logits = self.z_layer(self.output)

        probabilities = torch.sigmoid(logits).squeeze(-1)

        mask = (inputs["sentences"] != 0).float()

        output_dict = {}
        output_dict["probs"] = probabilities * mask
        output_dict["mask"] = mask
        predicted_rationale = (probabilities > 0.5).long()

        output_dict["predicted_rationale"] = predicted_rationale * mask.long()
        output_dict["prob_z"] = probabilities * mask

        return output_dict


class encoder(nn.Module):
    def __init__(self, masked_list = [0,101,102], output_dim = 2, dropout=0.1):
        
        super(encoder, self).__init__()

        """
        LEI BERT ENCODER
        Args:
            output_dim : number of labels to classify
            mask_list : a list of tokens that we want to pad out (e.g. SEP, CLS tokens)
                        when normalising the attentions. 
                        **** WARNING mask list is not used on the input ****
        Input:
            **inputs : dictionary with encode_plus outputs + retain_arg 
        Output:
            yhat : the predictions of the classifier
            rationale masks from the generator 
        """

        self.output_dim = output_dim        

        self.masked_list = masked_list
        self.dropout = dropout

        self.bert_generator = generator()

        self.enc_config = AutoConfig.from_pretrained(args["model"], output_attentions = True)   
        
        self.bert_encoder = AutoModel.from_pretrained(
                args["model"], 
                config=self.enc_config
            )

        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.bert_encoder.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

        self._reg_loss_lambda = 0.1
        self._reg_loss_mu = 2

    def forward(self, **inputs):

        ## pass through generator
        self.gen_dict = self.bert_generator(
            **inputs
        )

        self.mask = self.gen_dict["mask"]

        prob_z = self.gen_dict["probs"]

        self.sampler = torch.distributions.bernoulli.Bernoulli(probs=prob_z)

        if not self.training:
            
            sample_z = self.gen_dict["predicted_rationale"].float()
        
        else:
            sample_z = self.sampler.sample()

        self.sample_z = sample_z * self.mask

        reduced_input = inputs["sentences"] * self.sample_z.long() ## masking

        # reduced_input = self.select_tokens(inputs, self.sample_z) ##selection 

        ## to be used in loss function
        self.batch_length = max(inputs["lengths"])

        ## pass through encoder
        self.output, pooled_output, attention_weights = self.bert_encoder(reduced_input)


        # attention weights
        self.weights_or = attention_weights[-1]

	    # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        logits = self.output_layer(pooled_output)


        return logits, self.weights

    def _joint_rationale_objective(self, predicted_logits, actual_labels):

        """Losses taken from Jain et.al Learning
         to faithfully rationalize by construction"""

        # log_prob_z = self.sampler.log_prob(self.sample_z)  # (B, L)
        # log_prob_z_sum = (self.mask * log_prob_z).sum(-1)  # (B,)

        self.sample_z = self.sample_z[:,:self.batch_length]
        self.mask = self.mask[:,:self.batch_length]

        lasso_loss = self.sample_z.mean(1)
        censored_lasso_loss = nn.ReLU()(lasso_loss - args["rationale_length"])

        diff = (self.sample_z[:, 1:] - self.sample_z[:, :-1]).abs()
        mask_last = self.mask[:, :-1]
        fused_lasso_loss = diff.sum(-1) / mask_last.sum(-1)

        loss_sample = nn.CrossEntropyLoss(reduction = "none")(
            predicted_logits,
            actual_labels
        )


        base_loss = loss_sample
        generator_loss = (
                loss_sample.detach()
                + censored_lasso_loss * self._reg_loss_lambda
                + fused_lasso_loss * (self._reg_loss_mu * self._reg_loss_lambda)
            ) #* log_prob_z_sum

        return (base_loss + generator_loss).mean()

    def select_tokens(self, inputs, sample_z):
        sample_z_cpu = sample_z.cpu().data.numpy()
        assert len(inputs["sentences"]) == len(sample_z_cpu)

        new_document = []
        for jj, (doc, mask) in enumerate(zip(inputs["sentences"], sample_z_cpu)):
            doc = doc[:inputs["lengths"][jj]]
            mask = mask[: len(doc)]
            new_words = [doc[0]] + \
                 [doc[_i_] for _i_ in range(1,len(doc)-1) if mask[_i_] == 1] \
                     + [doc[-1]]

            new_document.append(new_words)

        max_doc = max([len(x) for x in new_document])

        new_padded = []

        for doc in new_document:

            cur_len = len(doc)
            len_diff = max_doc - cur_len

            new_padded.append(
                doc + [0]*len_diff
            )

        return torch.tensor(new_padded).to(device)
