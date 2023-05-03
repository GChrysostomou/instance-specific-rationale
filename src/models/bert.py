import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig,  AutoModelForQuestionAnswering
import json 
from src.models.bert_components import BertModelWrapper, mt5Wrapper, BertModelWrapper_zeroout, BertModelWrapper_noise, BertModelWrapper_attention
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
            ))
        

        # if if_multi == True:
        #     self.bert_config = AutoConfig.from_pretrained(
        #         args["multi_model"], 
        #         output_attentions = True)   
            
        #     self.wrapper = BertModelWrapper(
        #         AutoModel.from_pretrained(
        #             args["multi_model"], 
        #             config=self.bert_config
        #         ))


        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    
    def forward(self, **inputs):
    
        if "ig" not in inputs: inputs["ig"] = int(1)

        if "roberta" in args['model_abbreviation']:

            _, pooled_output, attention_weights = self.wrapper(
                inputs["input_ids"].to(device), 
                attention_mask = inputs["attention_mask"].to(device),
                token_type_ids = None,
                ig = inputs["ig"],
            )
        else: 
            _, pooled_output, attention_weights = self.wrapper(
                inputs["input_ids"].to(device), 
                attention_mask = inputs["attention_mask"].to(device),
                token_type_ids = inputs["token_type_ids"].to(device),
                ig = inputs["ig"],
            )
       
        # to retain gradients
        #self.weights_or = torch.tensor(attention_weights[-1], requires_grad=True)  # debug by cass
        self.weights_or = attention_weights[-1].clone().detach().requires_grad_(True)

        #  To copy construct from a tensor, it is recommended to use 
        # sourceTensor.clone().detach() or 
        # sourceTensor.clone().detach().requires_grad_(True), 
        # rather than torch.tensor(sourceTensor).
        
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


from src.models.modeling_mt5 import MT5Model 
from src.models.configuration_mt5 import MT5Config

class mt5(nn.Module):  # equal to "BertClassifier"
    def __init__(self, output_dim, if_output_attentions, dropout=0.1):
        
        super(mt5, self).__init__()

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

        
        self.dropout = dropout

        self.bert_config = MT5Config.from_pretrained(
            args["model"], 
            output_attentions = if_output_attentions)   
        
        self.wrapper = mt5Wrapper(
            MT5Model.from_pretrained(
                args["model"], 
                config=self.bert_config  # MT5Config
            ),
            )

        self.dropout = nn.Dropout(p = self.dropout)
        self.output_dim = output_dim
        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, output_dim) #self.wrapper.model.config.hidden_size
        # torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    
    def forward(self, **inputs):
    
        if "ig" not in inputs: inputs["ig"] = int(1)
        # mt5 forward ---> mt5wrapper forward the real model
        # the real model take input is the wrapper

        # forward content
        output = self.wrapper(
                input_ids = inputs["input_ids"].to(device), 
                attention_mask = inputs["attention_mask"].to(device),
                decoder_input_ids = inputs["decoder_input_ids"].to(device),
                decoder_attention_mask = inputs["decoder_attention_mask"].to(device),

                ig = inputs["ig"],
                return_dict = True,
        
                # head_mask:  Optional[torch.FloatTensor] = None,
                # decoder_head_mask:  Optional[torch.FloatTensor] = None,
                # cross_attn_head_mask: Optional[torch.Tensor] = None,
                # encoder_outputs:Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                # past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                # inputs_embeds:Optional[torch.Tensor] = None,
                # decoder_inputs_embeds: Optional[torch.Tensor] = None,
                # use_cache:Optional[bool] = None,
                # output_attentions: Optional[bool] = None,
                # output_hidden_states: Optional[bool] = None,
            )
        
            ### available in seq2seqmodelout: 
            # last_hidden_state=decoder_outputs.last_hidden_state,
            # past_key_values=decoder_outputs.past_key_values,
            # decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,



        
        decoder_lm_output = output[0] #  torch.Size([16, 5, 250112])
        decoder_sequence_output = output[1]
        attention_weights = output[2] # tuple, total 12 attention of torch.Size([16, 12, 5, 5])
        

        # label_input_indexd_dict[id_in_num_format] = label_input_id_index
        print(inputs.keys())
        num_index_dict = inputs['label_input_indexd_dict']

        selected_logits = []
        for i in range(len(num_index_dict)):
            
            if i ==0: selected_logits = decoder_lm_output[:, 0, int(torch.unique(num_index_dict[i][0]).item())].unsqueeze(1)
            else: 
                #print(' //////////// ', int(torch.unique(num_index_dict[i][0]).item()))
                selected_logit = decoder_lm_output[:, 0, int(torch.unique(num_index_dict[i][0]).item())].unsqueeze(1)
                #print(f"==>> selected_logit: {selected_logit}")
                selected_logits = torch.cat((selected_logits, selected_logit), 1)
        
      
        # to retain gradients
        #self.weights_or = torch.tensor(attention_weights[-1], requires_grad=True)  # debug by cass
        self.weights_or = attention_weights[-1].clone().detach().requires_grad_(True)
        # self.weights_or = torch.tensor(attention_weights.clone().detach()[-1], requires_grad=True)  # debug by cass

        if inputs["retain_gradient"]:
            # self.wrapper.model.embeddings.word_embeddings.weight.retain_grad() # different ---> softcomp: self.wrapper.word_embeds.retain_grad()
            self.wrapper.model.shared.weight.retain_grad()
            #self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad() # debug comment out by cass

	    # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or.mean(1)	


        return selected_logits, self.weights  

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

from transformers import MT5ForConditionalGeneration, T5Tokenizer


class mt5_backup(nn.Module):  # equal to "BertClassifier"
    def __init__(self, output_dim, if_output_attentions, dropout=0.1):
        
        super(mt5, self).__init__()
        
        self.dropout = dropout

        self.bert_config = MT5Config.from_pretrained(
            args["model"], 
            output_attentions = if_output_attentions)   
        
        self.wrapper = mt5Wrapper(
            MT5ForConditionalGeneration.from_pretrained(
                args["model"], 
                config=self.bert_config  # MT5Config
            ),
            )

        self.dropout = nn.Dropout(p = self.dropout)
        self.output_dim = output_dim
        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, output_dim) #self.wrapper.model.config.hidden_size
        # torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    
    def forward(self, **inputs):
    
        if "ig" not in inputs: inputs["ig"] = int(1)

        # mt5 forward ---> mt5wrapper forward the real model
        # the real model take input is the wrapper

        # forward content
        # output = decoder_outputs.last_hidden_state, 
                    # encoder_outputs.hidden_states, 
                    # encoder_outputs.attentions

        output = self.wrapper(
                input_ids = inputs["input_ids"].to(device), 
                attention_mask = inputs["attention_mask"].to(device),
                ig = inputs["ig"],
                return_dict = True,
                labels = inputs["labels"]
        
                # head_mask:  Optional[torch.FloatTensor] = None,
                # decoder_head_mask:  Optional[torch.FloatTensor] = None,
                # cross_attn_head_mask: Optional[torch.Tensor] = None,
                # encoder_outputs:Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                # past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                # inputs_embeds:Optional[torch.Tensor] = None,
                # decoder_inputs_embeds: Optional[torch.Tensor] = None,
                # use_cache:Optional[bool] = None,
                # output_attentions: Optional[bool] = None,
                # output_hidden_states: Optional[bool] = None,
            )
        
            ### available in seq2seqmodelout: 
            # last_hidden_state=decoder_outputs.last_hidden_state,
            # past_key_values=decoder_outputs.past_key_values,
            # decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,

        last_hidden_state = output[0][:, 0, :]  # torch.Size([16, 5, 768]) and the first token hiddent state
        encoder_hidden_states = output[1]
        attention_weights = output[2] # encoder_outputs attentions

       
        # to retain gradients
        #self.weights_or = torch.tensor(attention_weights[-1], requires_grad=True)  # debug by cass
        self.weights_or = attention_weights[-1].clone().detach().requires_grad_(True)
        # self.weights_or = torch.tensor(attention_weights.clone().detach()[-1], requires_grad=True)  # debug by cass

        if inputs["retain_gradient"]:
            # self.wrapper.model.embeddings.word_embeddings.weight.retain_grad() # different ---> softcomp: self.wrapper.word_embeds.retain_grad()
            self.wrapper.model.shared.weight.retain_grad()
            #self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad() # debug comment out by cass

	    # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or.mean(2)	

        logits = self.output_layer(last_hidden_state)

        

        return logits, self.weights  # after mean (2, 121, 3) become (2,3) 2 is batch_size, 3 num_of_label

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



class multi_bert(nn.Module):  # equal to "BertClassifier"
    def __init__(self, model_name, self_define_config, self_define_model, output_dim = 2, dropout=0.1):
        
        super(multi_bert, self).__init__()

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

        self.bert_config = self_define_config.from_pretrained(
                model_name, 
                output_attentions = True)   
            
        self.wrapper = multi_BertModelWrapper(
                self_define_model.from_pretrained(
                    model_name, 
                    config=self.bert_config
                ))


        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    
    def forward(self, **inputs):
    
        if "ig" not in inputs: inputs["ig"] = int(1)

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"].to(device), 
            attention_mask = inputs["attention_mask"].to(device),
            token_type_ids = inputs["token_type_ids"].to(device),
            ig = inputs["ig"],
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

        self.bert_config = AutoConfig.from_pretrained(args["model"], output_attentions = True)   
        
        self.wrapper = BertModelWrapper_zeroout(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config),
        ).to(device)
   
  
        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)
        # 这里OKAY

        _, pooled_output, attention_weights = self.wrapper(
            input_ids = inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"],
            rationale_mask = inputs["rationale_mask"], #.to(device),
            importance_scores = inputs["importance_scores"],
            faithful_method = inputs["faithful_method"],
            add_noise = inputs['add_noise'],
        )


        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        
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
    def __init__(self, std, output_dim = 2, dropout=0.1, tasc = None):
        
        super(BertClassifier_noise, self).__init__()

        self.output_dim = output_dim        
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(args["model"], output_attentions = True)

        self.std = std,   
        
        self.wrapper = BertModelWrapper_noise(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config),
                std=std,
        ).to(device)
        
  
        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)
        # 这里OKAY

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"].to(device), 
            attention_mask = inputs["attention_mask"].to(device),
            token_type_ids = inputs["token_type_ids"].to(device),
            ig = inputs["ig"],
            #rationale_mask = inputs["rationale_mask"].to(device),
            importance_scores = inputs["importance_scores"].to(device),
            faithful_method = inputs["faithful_method"],
            # std = inputs['std'],
            add_noise=inputs["add_noise"],
        )

        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        
        logits = self.output_layer(pooled_output.to(device))
        
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
   
        #self.tasc_mech = tasc
  
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
            rationale_mask = inputs["rationale_mask"].to(device),
            importance_scores = inputs["importance_scores"],
            faithful_method = inputs['faithful_method'],
            # std = inputs['std'],
        )
                # inputs["input_ids"])  
                # 这里OKAY， 但是第二次就不okay了
        # to retain gradients
        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        
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
