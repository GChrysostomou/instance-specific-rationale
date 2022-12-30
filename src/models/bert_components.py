from cmath import inf
from numpy import std
import torch
import torch.nn as nn
import json


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1.

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class aDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def bert_embeddings(bert_model, 
                    input_ids, 
                    position_ids = None, 
                    token_type_ids = None):

    """
    forward pass for the bert embeddings
    """

    if input_ids is not None:

        input_shape = input_ids.size()

    seq_length = input_shape[1]

    if position_ids is None:

        position_ids = torch.arange(512).expand((1, -1)).to(device)
        position_ids = position_ids[:, :seq_length]

    if token_type_ids is None:
    
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)  # +0.0000000001 by cass dabug

    embed = bert_model.embeddings.word_embeddings(input_ids.to(device))
    position_embeddings = bert_model.embeddings.position_embeddings(position_ids)
    token_type_embeddings = bert_model.embeddings.token_type_embeddings(token_type_ids)

    embeddings = embed + position_embeddings + token_type_embeddings
    embeddings = bert_model.embeddings.LayerNorm(embeddings)
    embeddings = bert_model.embeddings.dropout(embeddings)

    return embeddings, embed


class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
    
        super(BertModelWrapper, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        
    def forward(self, input_ids, attention_mask, token_type_ids, ig = int(1)):        

        embeddings, self.word_embeds = bert_embeddings(
            self.model, 
            input_ids.long(),
            position_ids = None, 
            token_type_ids = token_type_ids,
        )

        assert ig >= 0. and ig <= int(1), "IG ratio cannot be out of the range 0-1"
  
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers



        emb_temp = embeddings * ig
        encoder_outputs = self.model.encoder(
            emb_temp,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict
        )


        sequence_output = encoder_outputs[0]
        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions

# this version normalise
class BertModelWrapper_zeroout(nn.Module):
    
    def __init__(self, model):
    
        super(BertModelWrapper_zeroout, self).__init__()

        """
        BERT model wrapper
        """
        self.model = model
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                importance_scores, 
                add_noise,
                rationale_mask,
                faithful_method,
                ig = int(1),
                ):

        embeddings, self.word_embeds = bert_embeddings(
            self.model, 
            input_ids = input_ids, 
            position_ids = None, 
            token_type_ids = token_type_ids,
            )


        if type(ig) == int or type(ig) == float:
            assert ig >= 0. and ig <= int(1), "IG(Integrated Gradients: a postdoc explanations) ratio cannot be out of the range 0-1"
        else:
            ## else we need it to match the embeddings size for the KUMA mask
            ## therefore in this case ig is actually z from the KUMA model
            assert ig.size(0) == embeddings.size(0), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(1) == embeddings.size(1), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(2) == 1, "Rationale mask should be of size 1 in final dimension"
            ig = ig.float()
        
        
        if add_noise: # when for zero baseline
            #importance_scores[:,0] = 1 # preserve cls
            # importance_scores = torch.clip(importance_scores, min=-2).to(device)
            embeddings_3rd = embeddings.size(2)
            importance_scores = importance_scores.unsqueeze(2).repeat(1, 1, embeddings_3rd)

           
            if faithful_method == "soft_suff":
                        # the higher importance score, the more info for model
                        # the less perturbation, the less zero
                zeroout_mask = torch.bernoulli(importance_scores).to(device)

                embeddings = (embeddings * zeroout_mask).to(device)


            elif faithful_method == "soft_comp":
                zeroout_mask = torch.bernoulli(1-importance_scores).to(device)
                embeddings = embeddings * zeroout_mask

                rationale_mask_interleave = rationale_mask.repeat_interleave(embeddings.size()[2]).view(embeddings.shape)
                embeddings = rationale_mask_interleave * embeddings
        
            else: print(' something wrong !!!!!!!!!!!!!!!!!!!!!!!!')     


        else: pass # add noise = false


        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers

        encoder_outputs = self.model.encoder(
            embeddings * ig,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict
        )

        sequence_output = encoder_outputs[0]
        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions


class BertModelWrapper_zeroout_original_backup(nn.Module):
    
    def __init__(self, model):
    
        super(BertModelWrapper_zeroout, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        # self.importance_score = importance_score
        # self.faithful_method = faithful_method
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                faithful_method,
                importance_scores, 
                ig = int(1),
                ):     
        # 这里开始变成0, 一共三次， 第二次开始变0
        embeddings, self.word_embeds = bert_embeddings(
            self.model, 
            input_ids = input_ids, 
            position_ids = None, 
            token_type_ids = token_type_ids,
            )

        if type(ig) == int or type(ig) == float:
            assert ig >= 0. and ig <= int(1), "IG(Integrated Gradients: a postdoc explanations) ratio cannot be out of the range 0-1"
        else:
            ## else we need it to match the embeddings size for the KUMA mask
            ## therefore in this case ig is actually z from the KUMA model
            assert ig.size(0) == embeddings.size(0), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(1) == embeddings.size(1), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(2) == 1, "Rationale mask should be of size 1 in final dimension"
            ig = ig.float()
        

        # the fist token [CLS] not importance
        importance_scores[:,0] = 1e-4 
        importance_scores[torch.isinf(importance_scores)] = 1e-4
        importance_scores -= 1e-4 # modify by cass 1711
        importance_scores_min = importance_scores.min(1, keepdim=True)[0]
        importance_scores_max = importance_scores.max(1, keepdim=True)[0]
        importance_scores_nor = (importance_scores - importance_scores_min) / (importance_scores_max-importance_scores_min)
        
        
        # repeat
        importance_scores_nor_repeated = torch.repeat_interleave(torch.unsqueeze(importance_scores_nor, dim=-1), 
                                                        embeddings.shape[-1], dim=-1)
        
        if faithful_method == "soft_suff":
            # the higher importance score, the more info for model
            # the less perturbation, the less zero
            zeroout_mask = torch.bernoulli(importance_scores_nor_repeated).to(device)
            embeddings = embeddings * zeroout_mask
        elif faithful_method == "soft_comp":
            zeroout_mask = torch.bernoulli(1-importance_scores_nor_repeated).to(device)
            embeddings = embeddings * zeroout_mask
        else:
            pass

            
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers

        encoder_outputs = self.model.encoder(
            embeddings * ig,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict
        )

        sequence_output = encoder_outputs[0]


        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions




class GaussianNoise(nn.Module):

    def __init__(self, sigma=1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0)) # self.noise

    def forward(self, x, std):

        scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        sampled_noise = self.noise.expand(*x.size()).float().normal_(mean=0, std=std).to(device)
        scaled_sampled_noise =  sampled_noise * scale

        x = x + scaled_sampled_noise
        return x.to(device)


class BertModelWrapper_noise(nn.Module):
    
    def __init__(self, model, std):
    
        super(BertModelWrapper_noise, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        self.std = std
        
    def forward(self, input_ids, attention_mask, token_type_ids,
                importance_scores,
                rationale_mask,
                faithful_method,
                add_noise,
                ig = int(1),
                ):     
        
        embeddings, self.word_embeds = bert_embeddings(
            self.model.to(device), 
            input_ids = input_ids.to(device), 
            position_ids = None, 
            token_type_ids = token_type_ids.to(device),
            )

        add_noise = add_noise

        if type(ig) == int or type(ig) == float:
            assert ig >= 0. and ig <= int(1), "IG(Integrated Gradients: a postdoc explanations) ratio cannot be out of the range 0-1"

        else:
            ## else we need it to match the embeddings size for the KUMA mask
            ## therefore in this case ig is actually z from the KUMA model
            assert ig.size(0) == embeddings.size(0), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(1) == embeddings.size(1), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(2) == 1, "Rationale mask should be of size 1 in final dimension"
            ig = ig.float()



        ###############  combine with importance  #######
        if add_noise == False:
            pass
        else:

            #importance_scores[:,0] = 1 # preserve cls ?
            importance_score = importance_scores.clone().detach()


            if importance_score.size() != embeddings.size()[:2]: # embeddings.size()[1] is bigger than importance_score.size()[1]
                pad_x = torch.zeros((embeddings.size()[0], embeddings.size()[1]), 
                                    device=importance_score.device, dtype=importance_score.dtype)
                pad_x[:importance_score.size(0), :importance_score.size(1)] = importance_score
                importance_score = pad_x.clone().detach().to(device)

            
            if faithful_method == "soft_suff":
                for i in range(embeddings.size()[0]):
                    for k in range(embeddings.size()[1]):
                        importance_score_one_token = importance_score[i,k]
                        if importance_score_one_token != float('-inf'):
                            add_noise_fuc = GaussianNoise(sigma=(1-importance_score_one_token)) #is_relative_detach=True,  # importance_score_one_token is normalised to 0-1
                            embeddings[i,k,:] = add_noise_fuc(embeddings[i,k,:], std=self.std)

            elif faithful_method == "soft_comp":
                for i in range(embeddings.size()[0]):
                    for k in range(embeddings.size()[1]):
                        importance_score_one_token = importance_score[i,k]
                        if importance_score_one_token != float('-inf'):
                            add_noise_fuc = GaussianNoise(sigma=importance_score_one_token) #is_relative_detach=True, 
                            embeddings[i,k,:] = add_noise_fuc(embeddings[i,k,:], std=self.std)

                rationale_mask_interleave = rationale_mask.repeat_interleave(embeddings.size()[2]).view(embeddings.shape)
                embeddings = rationale_mask_interleave * embeddings
            else: pass # no changes to embeddings


        #attention_mask = importance_scores.detach().clone().to(device)

  
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers

        embeddings = embeddings * ig

        encoder_outputs = self.model.encoder(
            embeddings.to(device),
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict,
        )

        # quit()

        sequence_output = encoder_outputs[0].to(device)


        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions#.to(device)



class BertModelWrapper_attention(nn.Module):
    
    def __init__(self, model):
    
        super(BertModelWrapper_attention, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        # self.importance_score = importance_score
        # self.faithful_method = faithful_method
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                faithful_method,
                rationale_mask,
                importance_scores, # 进来的importance score就是指定了的某种method的score
                ig = int(1), add_noise=False,
                ):     
        
        embeddings, self.word_embeds = bert_embeddings(
            self.model, 
            input_ids = input_ids, 
            position_ids = None, 
            token_type_ids = token_type_ids,
            )

        add_noise = add_noise

        if type(ig) == int or type(ig) == float:

            assert ig >= 0. and ig <= int(1), "IG(Integrated Gradients: a postdoc explanations) ratio cannot be out of the range 0-1"

        else:
            ## else we need it to match the embeddings size for the KUMA mask
            ## therefore in this case ig is actually z from the KUMA model
            assert ig.size(0) == embeddings.size(0), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(1) == embeddings.size(1), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(2) == 1, "Rationale mask should be of size 1 in final dimension"
            ig = ig.float()

        ############### normalised to 0 to 1  #######
        if add_noise == False:
            pass
        else:
            if importance_scores.sum() ==  0: 
                pass
            else:
                importance_scores[:,0] = 1 


  
                if faithful_method == "soft_suff":
                    # the higher (lower --> 0) importance score, the more info for model
                    # the less perturbation, the less (more) masked value
                    # importance = 0 --> more masked --> more zero mask, lower value mask
                    attention_mask = importance_scores.detach().clone().to(device)
                elif faithful_method == "soft_comp":
                    attention_mask = (1-importance_scores).detach().clone().to(device)
                    rationale_mask_interleave = rationale_mask.repeat_interleave(embeddings.size()[2]).view(embeddings.shape)
                    embeddings = rationale_mask_interleave * embeddings
                else:
                    pass


        #attention_mask = importance_scores.detach().clone().to(device)

  
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers

        encoder_outputs = self.model.encoder(
            embeddings * ig,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict,
        )

        sequence_output = encoder_outputs[0]


        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions



