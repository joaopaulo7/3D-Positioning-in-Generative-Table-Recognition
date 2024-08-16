import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict 

import torch
from torch import nn, Tensor

from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.mbart.modeling_mbart import MBartPreTrainedModel, MBartForCausalLM, MBartDecoderWrapper, MBartDecoder, MBartDecoderLayer

from modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.utils import replace_return_docstrings

from configuration_dimbart import DiMBartConfig


_CONFIG_FOR_DOC = "DiMBartConfig"


def pickl_lambda():
        return 0

class DimPositionalEmbedding(nn.Module):
    """
    This module learns n sets positional embeddings up to n fixed maximum sizes.
    """

    def __init__(self, dim_counters, max_dim_lens, embeddings_d, dropout = 0.1):
        super().__init__()
        
        self.offset = 2
        self.embeddings_d = embeddings_d
        self.counter_dim = len(dim_counters)+1
        self.max_dim_lens = max_dim_lens
        self.max_dim_lens[0] += self.offset
        self.dropout = nn.Dropout(p=dropout/self.counter_dim)
        
        embeddings_list = []
        for i in range(len(max_dim_lens)):
            embeddings_list.append(nn.Embedding(self.max_dim_lens[i], self.embeddings_d))
        
        self.embeddings = nn.ModuleList(embeddings_list)
        
        self.counter_keys = defaultdict(pickl_lambda) 
        for i in range(self.counter_dim-1):
            for counter_key in dim_counters[i]:
                self.counter_keys[counter_key] = i+1

                
    def get_sequence_embeddings_map(self, input_ids:torch.Tensor, counters_state:List = None):
        seq_len = input_ids.shape[0]
        embeddings_map = [[self.max_dim_lens[j]-1 for i in range(seq_len)] for j in range(self.counter_dim)]
        
        if counters_state == None:
            counters = [0 for i in range(self.counter_dim)]
            counters[0] = self.offset
        else:
            counters = counters_state
        
        for i, token in enumerate(input_ids.tolist()):
            
            if token == 1:
                break
            
            counter = self.counter_keys[token]
            
            counters[counter] += 1
            
            #zeroes counters down the hierarchy
            for j in range(counter):
                counters[j] = 0
            
            #makes sure no counter ins above limit.
            carry = 0 
            for j in range(counter, self.counter_dim):
                if counters[j]+carry >= self.max_dim_lens[j]:
                    counters[j] = 0
                    carry = 1
                else:
                    counters[j] += carry
                    carry = 0
            
            #sets map
            for j in range(self.counter_dim):
                embeddings_map[j][i] = counters[j]
                
         
        return embeddings_map, counters.copy()
    
    
    def get_embeddings_sum(self, index_lists:List):
        embeddings_sum = torch.zeros((len(index_lists[0]),self.embeddings_d),
                                     device = self.embeddings[0].weight.device,
                                     dtype  = self.embeddings[0].weight.dtype)

        
        for i, index_list in enumerate(index_lists):
            partial_embedding = self.embeddings[i](torch.as_tensor(index_list, device = self.embeddings[0].weight.device))
            embeddings_sum = torch.add(embeddings_sum, self.dropout(partial_embedding))
        
        return embeddings_sum
    
    
    def forward(self, input_ids: torch.Tensor, counters_states:List = None):
        new_counters_states = []
        pos_embeddings = []
        
        for i, sequence in enumerate(input_ids):
            if counters_states == None:
                embeddings_map, new_counters_state = self.get_sequence_embeddings_map(sequence, None)
            else:
                embeddings_map, new_counters_state = self.get_sequence_embeddings_map(sequence, counters_states[i])
            
            pos_embeddings.append(self.get_embeddings_sum(embeddings_map))
            new_counters_states.append(new_counters_state)
           
        
        return torch.stack(pos_embeddings), new_counters_states


'''
Mostly copied from transformers.models.mbart.modeling_mbart.MBartForCausalLM.
  With a few changes as to configure and propagate counters_states changes
'''
class DiMBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MBartDecoderLayer`]

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: DiMBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = DimPositionalEmbedding(
            config.pos_counters,
            config.dim_max_position_embeddings,
            config.d_model,
            config.dropout
        )
        
        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        if past_key_values != None:
            counters_states = past_key_values[0][0].counters_states
        else:
            counters_states = None
            
        positions, counters_states = self.embed_positions(input_ids, counters_states)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if next_cache != None:
            next_cache[0][0].counters_states = counters_states
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    

# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderWrapper with MBart->DiMBart
class DiMBartDecoderWrapper(MBartDecoderWrapper):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config:DiMBartConfig):
        super(MBartPreTrainedModel, self).__init__(config)
        self.decoder = DiMBartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)



'''
Mostly copied from transformers.models.mbart.modeling_mbart.MBartForCausalLM.
  With a few changes as to propagate counters_states changes
'''
class DiMBartForCausalLM(MBartForCausalLM):

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super(MBartPreTrainedModel, self).__init__(config)
        self.model = DiMBartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        counters_states = past_key_values[0][0].counters_states 
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        
        reordered_past[0][0].counters_states = counters_states 
        return reordered_past
