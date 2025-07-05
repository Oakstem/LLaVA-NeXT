#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import ast
import re
import math

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.mm_utils import select_best_resolution # Import the helper from mm_utils

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        # Store image_grid_pinpoints and final_patch_division_size if available in config
        # These are usually part of data_args or model_cfg in scripts, need to ensure they are in self.config
        self.image_grid_pinpoints_config = getattr(config, "image_grid_pinpoints", None)
        # final_patch_division_size typically from processor.crop_size["height"]
        # This might need to be explicitly passed or inferred if not in main config
        self.final_patch_division_size_config = getattr(config, "final_patch_division_size", 224) # Example default


        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None, # List of [original_h, original_w]
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        atten_ids=[],
        pixel_coords_for_attention: Optional[List[Tuple[int, int]]] = None, # New parameter
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        ids_to_attend_pixels = []
        original_input_ids = input_ids # Save before potential modification
        # original_attention_mask = attention_mask # Keep a reference if needed

        if inputs_embeds is None:
            if images is not None and image_sizes is not None:
                (input_ids, position_ids, attention_mask_prepared, past_key_values, inputs_embeds, labels,
                 _image_features_ret, _user_prompt_features_ret, self.tokens_indexing) = \
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids=original_input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        labels=labels,
                        images=images,
                        modalities=modalities,
                        image_sizes=image_sizes
                    )
            else:
                # Unpack 6 values when images are not present (e.g., subsequent generation steps)
                # This path is taken when processing text-only inputs or during generation steps after the first one.
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    labels=labels,
                    images=images,
                    modalities=modalities,
                    image_sizes=image_sizes
                )

        final_ids_to_attend = kwargs.get("boost_positions" , None)

        # if inputs_embeds is not None:   # todo: uncomment once done testing, commenting out to get custom mask during generation too
        if inputs_embeds is None:
            input_embeds_shape = torch.Size([1, past_key_values[0][0].shape[2]+1, 1])
            input_device = past_key_values[0][0].device
            input_dtype = past_key_values[0][0].dtype
            # final_ids_to_attend = None
            # kwargs["boost_positions"] = None
        else:
            input_embeds_shape = inputs_embeds.shape
            input_device = inputs_embeds.device
            input_dtype = inputs_embeds.dtype
        # Only build custom mask if final_ids_to_attend is populated.
        # Otherwise, the attention_mask from prepare_inputs_labels_for_multimodal (which should be causal) or passed in, is used.
        if isinstance(final_ids_to_attend, dict):
            source_ids_to_attend = final_ids_to_attend.get("gaze_source", None)
            final_ids_to_attend = final_ids_to_attend.get("gaze_target", None)
        else:
            source_ids_to_attend = None
            
        if final_ids_to_attend:
            attention_mask = LlavaQwenForCausalLM._build_custom_attention_mask_static(
                input_embeds=inputs_embeds,
                inputs_embeds_shape=input_embeds_shape,
                ids_to_attend=final_ids_to_attend,
                tokens_indexing=self.tokens_indexing,
                device=input_device,
                dtype=input_dtype,
            )
        if source_ids_to_attend is not None:
            # If source_ids_to_attend is provided, we also build a mask for it.
            # This is useful for cases where we want to boost attention to specific tokens.
            source_attention_mask = LlavaQwenForCausalLM._build_custom_attention_mask_static(
                input_embeds=inputs_embeds,
                inputs_embeds_shape=input_embeds_shape,
                ids_to_attend=source_ids_to_attend,
                tokens_indexing=self.tokens_indexing,
                device=input_device,
                dtype=input_dtype,
            )
        else:
            source_attention_mask = None

        if inputs_embeds is None and attention_mask is not None:
            tokens_to_take = 1
            attention_mask = attention_mask[:, :, -tokens_to_take:, :]      # reduce only to the last query token
            if source_attention_mask is not None:
                source_attention_mask = source_attention_mask[:, :, -tokens_to_take:, :]
        # If final_ids_to_attend is empty, attention_mask remains as is.
        # It's assumed that if images were processed, prepare_inputs_labels_for_multimodal
        # would have set up a suitable (e.g., causal) attention_mask for inputs_embeds.
        # If inputs_embeds was passed directly, then the passed attention_mask is used.

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                source_attention_mask=source_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                boost_positions=kwargs.get("boost_positions", None),
                bias_strength=kwargs.get("bias_strength", None),
                tokens_indexing=self.tokens_indexing,
                query_indices=kwargs.get("query_indices", None),
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, image_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        result = super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return result, image_embeds

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    @staticmethod
    def _build_custom_attention_mask_static(input_embeds: Optional[torch.Tensor],
                                            inputs_embeds_shape: Tuple[int, ...],
                                            ids_to_attend: List[int],
                                            tokens_indexing: dict,
                                            device: torch.device,
                                            dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Builds a custom attention mask.
        The mask allows causal attention for all tokens.
        For the last token, it ONLY allows attention to `ids_to_attend`.
        """
        batch_size, seq_len, embed_size = inputs_embeds_shape

        # Initialize with causal mask properties
        mask = torch.full((seq_len, seq_len), float("0"), device=device, dtype=dtype)
        causal_indices = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        mask[causal_indices] = 1.

        if input_embeds is None:
             # lets mask all the image tokens
            mask[:, tokens_indexing['image'][0]] = 0.
        
        # Modify the last row for specific attention if ids_to_attend is provided
        if ids_to_attend and seq_len > 0:
            last_token_idx = seq_len - 1
            # mask[last_token_idx, :] = float("0")  # Disallow all by default for the last token

            valid_ids_to_attend = [idx for idx in ids_to_attend if 0 <= idx < seq_len]
            if valid_ids_to_attend:
                mask[last_token_idx, torch.tensor(valid_ids_to_attend, device=device, dtype=torch.long)] = 1.
        else:            
            k = 1

        # Reshape to [batch_size, 1, seq_len, seq_len] for broadcasting with attention heads
        # Qwen2 expects (batch_size, num_heads, query_length, kv_length) or (batch_size, 1, query_length, kv_length)
        # The original code produced [1,1,SL,SL]. We assume batch_size is handled by broadcasting if this is [1,1,SL,SL]
        # or we make it [B,1,SL,SL]
        return mask.unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
