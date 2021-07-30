import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig,  AutoModelForQuestionAnswering
import json 
from src.models.bert_components import BertModelWrapper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
            args = AttrDict(json.load(f))

class bert(nn.Module):
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

        self.bert_config = AutoConfig.from_pretrained(args["model"], output_attentions = True)   
        
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

        self.output, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"]
        )


        # to retain gradients
        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.weights_or.retain_grad()
            self.wrapper.word_embeds.retain_grad()
            # self.wrapper.model.embeddings.word_embeddings.weight.retain_grad()

	    # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	

        logits = self.output_layer(pooled_output)

        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            self.eval()
            self.zero_grad()

            inputs["ig"] = x
            
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


# class qa_bert(nn.Module):
#     def __init__(self, masked_list = [0,101,102], output_dim = None, dropout=0.1):
        
#         super(qa_bert, self).__init__()

#         """
#         QA bert
#         Args:
#             output_dim : number of labels to classify
#             mask_list : a list of tokens that we want to pad out (e.g. SEP, CLS tokens)
#                         when normalising the attentions. WARNING mask list is not used 
#                         on the input
#         Input:
#             input : the one hot encoded sequence representation, size [b_size, seq_len]
#             lengths : the lengths of the batched sequence, size [b_size, 1]
#             retain_gradient : retain the gradients for the embedding layer to obtain saliency scores
#         Output:
#             output : The concatenated encoded representations, size [b_size, seq_len, hidden_dim]
#             last_hidden : The encoded representation of the last step, size [b_size, hidden_dim]
#         """
      
#         self.bert_hidden_dim = 768
#         self.masked_list = masked_list
#         self.dropout = dropout

#         bert_model =  AutoModelForQuestionAnswering.from_pretrained(args["model"], 
#                                                             return_dict = True,
#                                                             output_attentions = True)      


#         self.bert_model = BertModelWrapperQA(bert_model)  

#     def forward(self, start_positions = None, end_positions = None, **inputs):

#         outputs = self.bert_model(
#                         input_ids = inputs["input_ids"], 
#                         token_type_ids = inputs["token_type_ids"],
#                         attention_mask = inputs["attention_mask"],
#                         salient_scores = inputs["salient_scores"]
#                         )



#         if inputs["retain_gradient"]:
            
#             self.weights_or.retain_grad()
#             self.bert_model.model.embeddings.word_embeddings.weight.retain_grad()


#         self.weights = outputs.attentions[-1][:, :, 0, :].mean(1)
#         # to retain gradients
#         self.weights_or = outputs.attentions[-1]

#         if self.bert_model.model.config.use_return_dict:del outputs["attentions"]
#         else: del outputs[-1]
        
#         if start_positions is not None and end_positions is not None:
            
#             if self.bert_model.model.config.use_return_dict: start_logits, end_logits = outputs.start_logits, outputs.end_logits
#             else: start_logits, end_logits = outputs[0], outputs[1]

#             # mask from mask_list used to remove SOS, EOS and PAD tokens
#             normalised_mask = torch.zeros_like(start_logits).bool()
        
#             for item in self.masked_list:
            
#                 normalised_mask += (inputs["input_ids"] == item).bool()
       
#             # mask unwanted tokens
#             start_logits = torch.masked_fill(start_logits, normalised_mask.to(device), float("-inf"))
#             end_logits = torch.masked_fill(end_logits, normalised_mask.to(device), float("-inf"))

#             start_logits = torch.softmax(start_logits, dim = -1)
#             end_logits = torch.softmax(end_logits, dim = -1)

#             total_loss = None

#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)

#             loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#             if self.bert_model.model.config.use_return_dict: outputs["loss"], outputs["start_logits"], outputs["end_logits"] = total_loss, start_logits, end_logits
#             else: outputs = [total_loss, start_logits, end_logits]
        
#         return outputs, self.weights

#     def normalise_scores(self, scores, sequence):
        
#         """
#         returns word-piece normalised scores
#         receives as input the scores {attention-weights, gradients} from bert and the sequence
#         the sequence is used to create a mask with default tokens masked (101,102,0)
#         which correspond to (CLS, SEP, PAD)
#         """

#         # mask from mask_list used to remove SOS, EOS and PAD tokens
#         self.normalised_mask = torch.zeros_like(scores).bool()
    
#         for item in self.masked_list:
        
#             self.normalised_mask += (sequence == item).bool()

#         # mask unwanted tokens
#         scores = torch.masked_fill(scores, self.normalised_mask.to(device), 0)
   
#         # return normalised word-piece scores       
#         return scores / scores.sum(-1, keepdim = True)



