import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig,  AutoModelForQuestionAnswering
import json 
from src.models.bert_components import BertModelWrapper, BertModelWrapper_zeroout, BertModelWrapper_noise, BertModelWrapper_attention
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
            args = AttrDict(json.load(f))

class bert(nn.Module):  # equal to "BertClassifier"
    def __init__(self, output_dim = 2, dropout=0.1):
        
        super(bert, self).__init__()

        """
        BERT FOR CLASSIFICATION
        Args:
            output_dim : number of labels to classify
            mask_list : a list of tokens that we want to pad out (e.g. SEP, CLS tokens)
                        when normalising the attentions. 
                        **** WARNING mask list is not used on the input ****
        Input:
            **inputs : dictionary with encode_plus outputs + salient scores if needed + retain_arg 
        Output:
            yhat : the predictions of the classifier
            attention weights : the attenion weights corresponding to the tokens from the last layer
        """

        self.output_dim = output_dim
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(
            args["model"], 
            output_attentions = True)   
        
        self.wrapper = BertModelWrapper(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config
            )
        )

        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    
    def forward(self, **inputs):
    
        if "ig" not in inputs: inputs["ig"] = int(1)

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"]
        )


        # to retain gradients
        self.weights_or = torch.tensor(attention_weights[-1], requires_grad=True)  # debug by cass
        #self.weights_or = attention_weights[-1].clone.detach().requires_grad_(True)#, requires_grad=True)

        # self.weights_or = torch.tensor(
        #     attention_weights.clone().detach()[-1], requires_grad=True)  # debug by cass

        if inputs["retain_gradient"]:
            
            # self.weights_or.retain_grad() 
            # self.wrapper.word_embeds.retain_grad()
            self.wrapper.model.embeddings.word_embeddings.weight.retain_grad() # different ---> softcomp: self.wrapper.word_embeds.retain_grad()
            #self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad() # debug comment out by cass

	    # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        logits = self.output_layer(pooled_output)

        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            self.eval()
            self.zero_grad()

            inputs["ig"] = x  # softcomp: inputs["ig"] = x.item()
            
            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward()

            #embedding gradients
            embed_grad = self.wrapper.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["input_ids"].long()]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.wrapper.model.embeddings.word_embeddings.weight[inputs["input_ids"].long()]

        ig = (attributions* em).sum(-1)
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig



# soft = zeroout
class BertClassifier_zeroout(nn.Module):
    def __init__(self, output_dim = 2, dropout=0.1, tasc = None):
        
        super(BertClassifier_zeroout, self).__init__()

        self.output_dim = output_dim        
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(
            args["model"], 
            output_attentions = True
        )   
        
        self.wrapper = BertModelWrapper_zeroout(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config),
            
        )
   
        self.tasc_mech = tasc
  
        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)
        # 这里OKAY

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"],
            tasc_mech = self.tasc_mech,
            importance_scores = inputs["importance_scores"],
            faithful_method = inputs["faithful_method"],
        )

        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        # print('after wrapper, the pooled_output: ', pooled_output)
        
        logits = self.output_layer(pooled_output)
        
        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            self.eval()
            self.zero_grad()

            inputs["ig"] = x.item()
            
            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward() #CUDA out of memory

            #embedding gradients
            embed_grad = self.wrapper.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["input_ids"].long()]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.wrapper.model.embeddings.word_embeddings.weight[inputs["input_ids"].long()]

        ig = torch.norm(attributions* em, dim = -1)
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig


class BertClassifier_noise(nn.Module):
    def __init__(self, output_dim = 2, dropout=0.1, tasc = None, std=0.5):
        
        super(BertClassifier_noise, self).__init__()

        self.output_dim = output_dim        
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(
            args["model"], 
            output_attentions = True
        )

        self.std = std,   
        
        self.wrapper = BertModelWrapper_noise(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config),
                std=std,
            
        )
        
        self.tasc_mech = tasc
  
        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)
        # 这里OKAY

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"],
            tasc_mech = self.tasc_mech,
            importance_scores = inputs["importance_scores"],
            faithful_method = inputs["faithful_method"],
            # std = inputs['std'],
            add_noise=inputs["add_noise"],
        )
        # print(' HERE, finish one forward/one wrapper func')
        # print('inputs["input_ids"] inside classifier and AFTER wrapper   --->', 
                # inputs["input_ids"])  
                # 这里OKAY， 但是第二次就不okay了
        # to retain gradients
        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        # print('after wrapper, the pooled_output: ', pooled_output)
        
        logits = self.output_layer(pooled_output)
        
        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            self.eval()
            self.zero_grad()

            inputs["ig"] = x.item()
            
            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward() #CUDA out of memory

            #embedding gradients
            embed_grad = self.wrapper.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["input_ids"].long()]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.wrapper.model.embeddings.word_embeddings.weight[inputs["input_ids"].long()]

        ig = torch.norm(attributions* em, dim = -1)
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig



class BertClassifier_attention(nn.Module):
    def __init__(self, output_dim = 2, dropout=0.1, tasc = None):
        
        super(BertClassifier_attention, self).__init__()

        self.output_dim = output_dim        
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(
            args["model"], 
            output_attentions = True
        )   
        
        self.wrapper = BertModelWrapper_attention(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config),
            
        )
   
        self.tasc_mech = tasc
  
        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)
        # 这里OKAY

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"],
            tasc_mech = self.tasc_mech,
            importance_scores = inputs["importance_scores"],
            faithful_method = inputs['faithful_method'],
            # std = inputs['std'],
        )
        # print(' HERE, finish one forward/one wrapper func')
        # print('inputs["input_ids"] inside classifier and AFTER wrapper   --->', 
                # inputs["input_ids"])  
                # 这里OKAY， 但是第二次就不okay了
        # to retain gradients
        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        # print('after wrapper, the pooled_output: ', pooled_output)
        
        logits = self.output_layer(pooled_output)
        
        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            self.eval()
            self.zero_grad()

            inputs["ig"] = x.item()
            
            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward() #CUDA out of memory

            #embedding gradients
            embed_grad = self.wrapper.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["input_ids"].long()]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.wrapper.model.embeddings.word_embeddings.weight[inputs["input_ids"].long()]

        ig = torch.norm(attributions* em, dim = -1)
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig
