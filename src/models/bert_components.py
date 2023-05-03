from cmath import inf
from numpy import std
import numpy
import torch
import torch.nn as nn
import json
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    if args['model_abbreviation'] == "flaubert": embed = bert_model.embeddings(input_ids.to(device))
    else: embed = bert_model.embeddings.word_embeddings(input_ids.to(device))

    if args['model_abbreviation'] == "deberta": 
        position_embeddings = torch.empty(embed.size()).to(device)
        token_type_embeddings = torch.empty(embed.size()).to(device)
    elif args['model_abbreviation'] == "flaubert": 
        position_embeddings = bert_model.position_embeddings(position_ids)
        token_type_embeddings = bert_model.embeddings(token_type_ids)
    else: 
        position_embeddings = bert_model.embeddings.position_embeddings(position_ids)
        token_type_embeddings = bert_model.embeddings.token_type_embeddings(token_type_ids)

    if token_type_ids is None:embeddings = embed + position_embeddings
    else:embeddings = embed + position_embeddings + token_type_embeddings

    if args['model_abbreviation'] == "flaubert": embeddings = bert_model.layer_norm_emb(embeddings)
    else: embeddings = bert_model.embeddings.LayerNorm(embeddings)

    #print(embeddings)
    if args['model_abbreviation'] == "flaubert": 
        #print(bert_model)
        #print(bert_model.dropout())
        dropout_layer = nn.Dropout(p=0.1)
        embeddings_temp = dropout_layer(embeddings)
        return embeddings_temp, embed
    
    else: 
        embeddings = bert_model.embeddings.dropout(embeddings)

        return embeddings, embed

def m2m_embeddings(bert_model, 
                    input_ids, 
                    position_ids = None, 
                    token_type_ids = None):

    """
    forward pass for the bert embeddings
    """

    if input_ids is not None: input_shape = input_ids.size()

    seq_length = input_shape[1]

    if position_ids is None:

        position_ids = torch.arange(512).expand((1, -1)).to(device)
        position_ids = position_ids[:, :seq_length]

    if token_type_ids is None:
    
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)  # +0.0000000001 by cass dabug

    # embed = bert_model.get_input_embeddings(input_ids.to(device))
    embed = bert_model.get_input_embeddings()
    position_embeddings = bert_model.embeddings.position_embeddings(position_ids)
    token_type_embeddings = bert_model.embeddings.token_type_embeddings(token_type_ids)

    embeddings = embed + position_embeddings + token_type_embeddings
    embeddings = bert_model.embeddings.LayerNorm(embeddings)
    embeddings = bert_model.embeddings.dropout(embeddings)

    return embeddings, embed

class multi_BertModelWrapper(nn.Module):
    
    def __init__(self, model):
    
        super(multi_BertModelWrapper, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.activation = nn.Tanh()
        
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
        head_mask =  [None] * self.model.config.num_hidden_layers
        emb_temp = embeddings * ig
        # encoder_outputs = self.model.encoder(
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     inputs_embeds = emb_temp,
        #     output_attentions=self.model.config.output_attentions,
        #     output_hidden_states=self.model.config.output_attentions,
        #     return_dict=self.model.config.return_dict
        # )
        # if 'xlm' in args['multi_model_name']:
        #     # print(' 000000000000000000000 )))))')
        #     # print(self.weight)
        #     encoder_outputs = self.model.encoder(
        #         input_ids,
        #         attention_mask=attention_mask,
        #         #head_mask=head_mask,
        #         #inputs_embeds=inputs_embeds,
        #         output_attentions=self.model.config.output_attentions,
        #         output_hidden_states=self.model.config.output_attentions,
        #         return_dict=self.model.config.return_dict
        #     )

        # else:
        encoder_outputs = self.model.encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    #head_mask=head_mask,
                    #inputs_embeds=inputs_embeds,
                    output_attentions=self.model.config.output_attentions,
                    output_hidden_states=self.model.config.output_attentions,
                    return_dict=self.model.config.return_dict
                )

        sequence_output = encoder_outputs[0]
        attentions = encoder_outputs[2]

        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        #pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output




class mT5Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

import copy, warnings
from src.models.modeling_mt5 import MT5Model,MT5Stack
from src.models.configuration_mt5 import MT5Config
from transformers.modeling_outputs import BaseModelOutput

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

from typing import Optional, Tuple, Union
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, T5Tokenizer



def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In MT5 it is usually set to the pad_token_id."
            " See MT5 docs for more information"
        )

        # shift inputs to the right
        # if is_torch_fx_proxy(input_ids):
        #     # Item assignment is not supported natively for proxies.
        #     shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        #     shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        # else:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

class mt5Wrapper_backup(nn.Module):
    r"""
    feed in model MT5ForConditionalGeneration
    ```"""
    def __init__(self, model):
        super(mt5Wrapper, self).__init__()

        self.model = model  # a pretrained load model MT5Model.from_pretrained(args["model"], config=self.bert_config),
        self.config = MT5Config.from_pretrained(args["model"], output_attentions = True)

        self.shared = nn.Embedding(self.config.vocab_size, self.config.d_model)

        #self.shared = nn.Embedding(250112, 768)  ## by cass

        encoder_config = MT5Config.from_pretrained(args["model"], output_attentions = True)    # copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = MT5Config.from_pretrained(args["model"])  
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = decoder_config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        # # Initialize weights and apply final processing
        # self.post_init()


    # Copied from transformers.models.t5.modeling_t5.T5Model.get_input_embeddings
    def get_input_embeddings(self):
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5Model.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_encoder
    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_decoder
    def get_decoder(self):
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5Model._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        ig,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )-> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = _shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
           
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )





class mt5Wrapper(nn.Module):
    r"""
    Examples:
    ```python
    >>> from transformers import MT5Model, AutoTokenizer
    >>> model = MT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> labels = tokenizer(text_target=summary, return_tensors="pt")
    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    # model_type = "mt5"
    # config_class = MT5Config
    # _keys_to_ignore_on_load_missing = [
    #     r"encoder.embed_tokens.weight",
    #     r"decoder.embed_tokens.weight",
    #     r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    # ]
    # _keys_to_ignore_on_save = [
    #     r"encoder.embed_tokens.weight",
    #     r"decoder.embed_tokens.weight",
    # ]
    # _keys_to_ignore_on_load_unexpected = [
    #     r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    # ]
    

    # Copied from transformers.models.t5.modeling_t5.T5Model.__init__ with T5->MT5
    def __init__(self, model):
        super(mt5Wrapper, self).__init__()

        self.model = model  # a pretrained load model MT5Model.from_pretrained(args["model"], config=self.bert_config),
        self.config = MT5Config.from_pretrained(args["model"], output_attentions = True)

        self.shared = nn.Embedding(self.config.vocab_size, self.config.d_model)

        #self.shared = nn.Embedding(250112, 768)  ## by cass

        encoder_config = MT5Config.from_pretrained(args["model"], output_attentions = True)    # copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = MT5Config.from_pretrained(args["model"])  
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = decoder_config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # # Initialize weights and apply final processing
        # self.post_init()


    # Copied from transformers.models.t5.modeling_t5.T5Model.get_input_embeddings
    def get_input_embeddings(self):
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5Model.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_encoder
    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_decoder
    def get_decoder(self):
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5Model._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,  #: Optional[torch.LongTensor] = None,
        #label_ids,
        attention_mask,  #: Optional[torch.FloatTensor] = None,
        ig,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask:  Optional[torch.FloatTensor] = None,
        decoder_head_mask:  Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs:Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds:Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache:Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, MT5Model
        >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
        >>> model = MT5Model.from_pretrained("mt5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for MT5Model.
        >>> # This is not needed for torch's MT5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)
        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = False # if use_cache is None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            '''
            return 
            BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
            
            '''
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            '''
            BaseModelOutput return
                last_hidden_state: torch.FloatTensor = None
                hidden_states: Optional[Tuple[torch.FloatTensor]] = None
                attentions: Optional[Tuple[torch.FloatTensor]] = None
            '''

        hidden_states = encoder_outputs[0] # last hiddent state

        '''     
        encoder  output is from MT5Stack  ===> only got 
        1. hidden_states torch.Size([16, 5, 768])
        2. all_attentions of tuple 12 torch.Size([16, 12, 5, 5]) torch.Size([16, 12, 5, 5])
        '''

        # Decode
        decoder_outputs = self.decoder(
            
            input_ids=decoder_input_ids,
            
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        '''     
        decoder output is from MT5Stack  ===> only got hidden_states  torch.Size([16, 5, 768])
        which is        
        return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=present_key_value_states,
                    hidden_states=all_hidden_states,
                    attentions=all_attentions,
                    cross_attentions=all_cross_attentions,
                )
        '''
        if not return_dict:
            output = decoder_outputs + encoder_outputs  # output is a tuple
            return output

        #decoder_last_hidden_state = decoder_outputs.last_hidden_state

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.config.d_model**-0.5)
        lm_logits = self.lm_head(sequence_output)
        
        return lm_logits, sequence_output, encoder_outputs.attentions



class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
    
        super(BertModelWrapper, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        
    def forward(self, input_ids, attention_mask, token_type_ids, ig = int(1)):       
        #print(self.model)

        if args['model_abbreviation'] == "xlm_roberta":

            token_type_ids = None

        
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
        #print(f"==>> attentions len : {len(attentions)}  size --> {attentions[0].size()}")
        # pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None
        
        try: 
            pooled_output = self.model.pooler(sequence_output) 
        except:  pooled_output = sequence_output
        # sequence_output already in shape [batch size, num of label]
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

        
    #if the more importance, keep more in suff and less in comp, less zero in suff and more in comp
        embeddings_3rd = embeddings.size(2)
        importance_scores = importance_scores.unsqueeze(2).repeat(1, 1, embeddings_3rd)
        if torch.sum(torch.isnan(importance_scores)) > 2: # for fixed0, generate a zero ones
            importance_scores = torch.zeros(importance_scores.size())
            

        if faithful_method == "soft_suff":
            try: zeroout_mask = torch.bernoulli(1-importance_scores).to(device)
            except: 
                importance_scores = torch.zeros(embeddings.size())
                zeroout_mask = torch.bernoulli(1-importance_scores).to(device)

            embeddings = embeddings * zeroout_mask



        elif faithful_method == "soft_comp":
            #print(importance_scores)
            try: zeroout_mask = torch.bernoulli(importance_scores).to(device)
            except: 
                importance_scores = torch.zeros(embeddings.size())
                zeroout_mask = torch.bernoulli(importance_scores).to(device)
            
            embeddings = embeddings * zeroout_mask

            # rationale_mask_interleave = rationale_mask.repeat_interleave(embeddings.size()[2]).view(embeddings.shape)
            # embeddings = rationale_mask_interleave * embeddings

    
        else: print(' something wrong !!!!!!!!!!!!!!!!!!!!!!!!')     


    #else: pass # add noise = false


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



# sigma = importance scores
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
        self.model = model
        self.std = std
        
    def forward(self, input_ids, attention_mask, token_type_ids,
                importance_scores,
                #rationale_mask, # not using it
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
                        if importance_score_one_token != 0:
                            add_noise_fuc = GaussianNoise(sigma=(1-importance_score_one_token)) #is_relative_detach=True,  # importance_score_one_token is normalised to 0-1
                            embeddings[i,k,:] = add_noise_fuc(embeddings[i,k,:], std=self.std)

            elif faithful_method == "soft_comp":
                for i in range(embeddings.size()[0]):
                    for k in range(embeddings.size()[1]):
                        importance_score_one_token = importance_score[i,k]
                        if importance_score_one_token != 0:
                            add_noise_fuc = GaussianNoise(sigma=importance_score_one_token) #is_relative_detach=True, 
                            embeddings[i,k,:] = add_noise_fuc(embeddings[i,k,:], std=self.std)

                #############suff 在进来前处理掉了, id 直接遮掉了
                # rationale_mask_interleave = rationale_mask.repeat_interleave(embeddings.size()[2]).view(embeddings.shape)
                # embeddings = rationale_mask_interleave * embeddings

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



