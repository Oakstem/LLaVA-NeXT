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


from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import ast
import re
import math
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput, GenerationMixin

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config        #, Qwen2Model, Qwen2ForCausalLM
from qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from llava.mm_utils import select_best_resolution # Import the helper from mm_utils
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.attention_mask_visualizer import visualize_attention_mask_step
from gazefollow.focus_loss_utils import locate_focus_start_index, prepare_focus_phrase_sequences
from gazefollow.roi_contrastive_utils import (
    ROIContrastivePreviewBuffer,
    build_bbox_patch_indices,
    build_circular_roi_patch_indices,
    build_roi_position_vector,
    roi_debug_log,
)

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


INSERT_EMBED_TOKEN_ID = 716

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
        roi_dim = int(getattr(config, "roi_contrastive_dim", 512))
        roi_pos_dim = int(getattr(config, "roi_pos_embed_dim", 64))
        self.roi_text_projector = nn.Linear(config.hidden_size, roi_dim, bias=False)
        self.roi_vision_projector = nn.Linear(config.hidden_size, roi_dim, bias=False)
        self.roi_position_mlp = nn.Sequential(
            nn.Linear(6, roi_pos_dim, bias=True),
            nn.GELU(),
            nn.Linear(roi_pos_dim, roi_dim, bias=False),
        )
        self._roi_contrastive_step = 0
        self._roi_contrastive_stats: Optional[Dict[str, torch.Tensor]] = None
        self._roi_contrastive_warned_small_pair_count = False
        self.latest_person_mask_repr: Optional[torch.Tensor] = None
        self.latest_attention_mask_snapshot: Optional[Dict[str, Any]] = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _build_tokens_indexing_from_ids(
        self, reference_input_ids: torch.LongTensor, seq_len: int
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Approximate tokens_indexing metadata from raw input IDs when image processing is skipped.
        """
        if reference_input_ids.dim() == 1:
            reference_input_ids = reference_input_ids.unsqueeze(0)

        tokens_indexing: Dict[str, List[torch.Tensor]] = {"image": [], "text": []}
        for im_sample_ids in reference_input_ids:
            image_inds = (im_sample_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
            text_inds = (im_sample_ids != IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
            tokens_indexing["image"].append(image_inds)
            tokens_indexing["text"].append(text_inds)

        special_inds = (reference_input_ids[-1] == INSERT_EMBED_TOKEN_ID).nonzero(as_tuple=False).flatten()
        if special_inds.numel() > 0:
            special_inds = torch.unique(special_inds)
            special_inds = special_inds[(special_inds >= 0) & (special_inds < seq_len)]
            insert_positions = {"source": [], "target": []}
            for idx, value in enumerate(special_inds.tolist()):
                if idx % 2 == 0:
                    insert_positions["source"].append(int(value))
                else:
                    insert_positions["target"].append(int(value))
            tokens_indexing["insert_embd"] = insert_positions
        else:
            tokens_indexing["insert_embd"] = None

        return tokens_indexing

    @staticmethod
    def _insert_positions_nonempty(positions: Optional[Any]) -> bool:
        if positions is None:
            return False
        if torch.is_tensor(positions):
            return positions.numel() > 0
        if isinstance(positions, np.ndarray):
            return positions.size > 0
        if isinstance(positions, (list, tuple)):
            return len(positions) > 0
        return True

    def _maybe_share_target_insert_positions(
        self,
        tokens_indexing: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(tokens_indexing, dict):
            return
        insert_positions = tokens_indexing.get("insert_embd")
        if not isinstance(insert_positions, dict):
            return
        target_positions = insert_positions.get("target")
        if not self._insert_positions_nonempty(target_positions):
            return
        insert_positions["source"] = target_positions

    def _apply_mask_embedding_logic(
        self,
        inputs_embeds: Optional[torch.Tensor],
        boost_positions: Optional[Dict[str, List[int]]],
        kwargs: Dict[str, Any],
    ) -> bool:
        """Apply mask embedding injections into the token representations if requested."""
        self.latest_person_mask_repr = None
        if inputs_embeds is None or not isinstance(boost_positions, dict):
            return False

        mask_hidden_state_repr = kwargs.get("target_mask_embedding")
        if mask_hidden_state_repr is None:
            return False

        tokens_indexing = getattr(self, "tokens_indexing", None)
        if not isinstance(tokens_indexing, dict):
            return False

        use_target_insert_override = bool(kwargs.get("use_target_insert_indices_for_source", False))
        if use_target_insert_override:
            self._maybe_share_target_insert_positions(tokens_indexing)

        insert_positions = tokens_indexing.get("insert_embd")
        if not isinstance(insert_positions, dict):
            return False

        def _normalize_positions(values):
            if values is None:
                return []
            if isinstance(values, torch.Tensor):
                return [int(val) for val in values.reshape(-1).tolist()]
            if isinstance(values, np.ndarray):
                return [int(val) for val in values.reshape(-1).tolist()]
            if isinstance(values, (list, tuple)):
                return [int(val) for val in values]
            return [int(values)]

        source_insert_targets = insert_positions.get("source")
        target_insert_targets = insert_positions.get("target")

        if isinstance(mask_hidden_state_repr, dict):
            mask_hidden_state_source = mask_hidden_state_repr.get("source")
            mask_hidden_state_target = mask_hidden_state_repr.get("target")
        else:
            mask_hidden_state_source = mask_hidden_state_repr
            mask_hidden_state_target = mask_hidden_state_repr

        target_attn_mask_indices = boost_positions.get("gaze_target")
        person_attn_mask_indices = boost_positions.get("gaze_source")
        if use_target_insert_override and target_attn_mask_indices is not None:
            person_attn_mask_indices = target_attn_mask_indices
            if mask_hidden_state_target is not None:
                mask_hidden_state_source = mask_hidden_state_target

        repr_layer_idx = kwargs.get("repr_layer_idx", None)
        use_layer_injection = repr_layer_idx is not None and insert_positions is not None
        layer_injection_data = None
        updated = False

        has_source_tokens = (
            person_attn_mask_indices is not None
            and len(person_attn_mask_indices) > 0
            and source_insert_targets
        )
        has_target_tokens = (
            target_attn_mask_indices is not None
            and len(target_attn_mask_indices) > 0
            and target_insert_targets
        )

        if mask_hidden_state_source is not None and has_source_tokens:
            person_mask_repr = mask_hidden_state_source[person_attn_mask_indices, :].mean(dim=0)
            person_mask_repr = person_mask_repr.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            if use_layer_injection:
                layer_injection_data = layer_injection_data or {}
                layer_injection_data["source"] = {
                    "positions": torch.as_tensor(
                        source_insert_targets, device=inputs_embeds.device, dtype=torch.long
                    ),
                    "embedding": person_mask_repr,
                    "inject_layer_idx": repr_layer_idx['inject']
                }
            elif source_insert_targets is not None:
                inputs_embeds[:, source_insert_targets, :] = person_mask_repr

            gaze_source_targets = boost_positions.get("gaze_source")
            if gaze_source_targets is None:
                boost_positions["gaze_source"] = []
                gaze_source_targets = boost_positions["gaze_source"]
            elif not isinstance(gaze_source_targets, list):
                if isinstance(gaze_source_targets, torch.Tensor):
                    gaze_source_targets = gaze_source_targets.reshape(-1).tolist()
                elif isinstance(gaze_source_targets, np.ndarray):
                    gaze_source_targets = gaze_source_targets.reshape(-1).tolist()
                else:
                    gaze_source_targets = [gaze_source_targets]
                gaze_source_targets = [int(val) for val in gaze_source_targets]
                boost_positions["gaze_source"] = gaze_source_targets

            gaze_source_targets += _normalize_positions(source_insert_targets)
            self.latest_person_mask_repr = person_mask_repr.detach()
            updated = True

        if mask_hidden_state_target is not None and has_target_tokens:
            target_mask_embedding = mask_hidden_state_target[target_attn_mask_indices, :].mean(dim=0)
            target_mask_embedding = target_mask_embedding.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            if use_layer_injection:
                layer_injection_data = layer_injection_data or {}
                layer_injection_data["target"] = {
                    "positions": torch.as_tensor(
                        target_insert_targets, device=inputs_embeds.device, dtype=torch.long
                    ),
                    "embedding": target_mask_embedding,
                    "inject_layer_idx": repr_layer_idx['inject']
                }
            elif target_insert_targets:
                inputs_embeds[:, target_insert_targets, :] = target_mask_embedding

            updated = True

        if use_layer_injection and layer_injection_data:
            kwargs["repr_injection"] = layer_injection_data
            updated = True

        return updated

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
        self._roi_contrastive_stats = None
        self.latest_person_mask_repr = None
        ids_to_attend_pixels = []
        original_input_ids = input_ids # Save before potential modification
        # original_attention_mask = attention_mask # Keep a reference if needed
        final_ids_to_attend = kwargs.get("boost_positions" , None)
        mask_logic_applied = False
        image_features_ret = None
        image_token_filter_indices = kwargs.pop("image_token_filter_indices", None)
        attention_mask_viz_config = kwargs.pop("attention_mask_viz", None)
        roi_gaze_xy = kwargs.pop("roi_gaze_xy", None)
        roi_gaze_valid = kwargs.pop("roi_gaze_valid", None)
        roi_candidate_boxes = kwargs.pop("roi_candidate_boxes", None)
        roi_candidate_is_positive = kwargs.pop("roi_candidate_is_positive", None)
        roi_candidate_valid = kwargs.pop("roi_candidate_valid", None)
        roi_contrastive_enabled = bool(getattr(self.config, "roi_contrastive_enable", False))
        roi_candidate_inputs_ready = (
            roi_candidate_boxes is not None
            and roi_candidate_is_positive is not None
            and roi_candidate_valid is not None
        )
        roi_requires_hidden_states = (
            roi_contrastive_enabled
            and self.training
            and not dpo_forward
            and images is not None
            and labels is not None
            and (
                (roi_gaze_xy is not None and roi_gaze_valid is not None)
                or roi_candidate_inputs_ready
            )
        )
        forced_output_hidden_states = roi_requires_hidden_states and not bool(output_hidden_states)
        if forced_output_hidden_states:
            output_hidden_states = True

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
                        image_sizes=image_sizes,
                        image_token_filter_indices=image_token_filter_indices,
                    )
                image_features_ret = _image_features_ret
                if kwargs.get("use_target_insert_indices_for_source"):
                    self._maybe_share_target_insert_positions(self.tokens_indexing)

                if kwargs.get("target_mask_embedding", None) is not None:
                    mask_logic_applied = self._apply_mask_embedding_logic(
                        inputs_embeds=inputs_embeds,
                        boost_positions=final_ids_to_attend,
                        kwargs=kwargs,
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
                        image_sizes=image_sizes,
                        image_token_filter_indices=image_token_filter_indices,
                )   

            # if kwargs.get("target_tokens", 0) == 2:
            #     if kwargs.get("target_mask_embedding", None) is not None:
            #         mask_embedding = kwargs["target_mask_embedding"]
            #         input_masks = kwargs.get("input_masks", {})
            #         target_attn_mask_indices = input_masks['target_mask'].flatten()
            #         person_attn_mask_indices = input_masks['person_mask'].flatten() 
            #         target_attn_mask_indices = np.where(target_attn_mask_indices > 0)[0]
            #         person_attn_mask_indices = np.where(person_attn_mask_indices > 0)[0]
            #         target_mask_embedding = mask_embedding[person_attn_mask_indices, :].mean(dim=0)
            #         inputs_embeds = target_mask_embedding.unsqueeze(0).expand(1, -1, -1)
            #         input_ids = None
        # if inputs_embeds is not None:   # todo: uncomment once done testing, commenting out to get custom mask during generation too
        # kwargs["repr_injection"] = None
        if inputs_embeds is None and past_key_values is None and input_ids is not None:
            # No modality inputs were provided for the first step; fall back to token embeddings.
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        if inputs_embeds is None: # or kwargs.get("target_tokens", 0) == 2:
            input_embeds_shape = torch.Size([1, past_key_values[0][0].shape[2]+1, 1])
            input_device = past_key_values[0][0].device
            input_dtype = past_key_values[0][0].dtype
            # final_ids_to_attend = None
            # kwargs["boost_positions"] = None
        else:
            input_embeds_shape = inputs_embeds.shape
            input_device = inputs_embeds.device
            input_dtype = inputs_embeds.dtype

        if getattr(self, "tokens_indexing", None) is None or not kwargs.get("include_image_inputs", False):
            try:
                self.tokens_indexing = self._build_tokens_indexing_from_ids(
                    original_input_ids, input_embeds_shape[1]
                )
                if kwargs.get("use_target_insert_indices_for_source"):
                    self._maybe_share_target_insert_positions(self.tokens_indexing)
            except Exception as exc:
                pass
        if not mask_logic_applied and kwargs.get("target_mask_embedding", None) is not None:
            mask_logic_applied = self._apply_mask_embedding_logic(
                inputs_embeds=inputs_embeds,
                boost_positions=final_ids_to_attend,
                kwargs=kwargs,
            )
        # Only build custom mask if final_ids_to_attend is populated and token indexing information is available.
        tokens_indexing = getattr(self, "tokens_indexing", None)
        image_index_list = None
        if isinstance(tokens_indexing, dict) and tokens_indexing.get("insert_embd", None) is not None:
            image_index_list = tokens_indexing.get("image")
            source = tokens_indexing["insert_embd"].get("source", None)
            has_gaze_source_indices = source is not None and len(source) > 0
            target = tokens_indexing["insert_embd"].get("target", None)
            has_gaze_target_indices = target is not None and len(target) > 0
        has_image_indices = (
            isinstance(image_index_list, list)
            and len(image_index_list) > 0
            and isinstance(image_index_list[0], torch.Tensor)
            and image_index_list[0].numel() > 0
        )
        can_build_custom_mask = has_image_indices
        mask_to_visualize = None
        if isinstance(final_ids_to_attend, dict):
            source_ids_to_attend = final_ids_to_attend.get("gaze_source", None)
            final_ids_to_attend = final_ids_to_attend.get("gaze_target", None)
        else:
            source_ids_to_attend = None

        # Gaze Target attention mask             
        if can_build_custom_mask and final_ids_to_attend is not None and has_gaze_target_indices:
            attention_mask = LlavaQwenForCausalLM._build_custom_attention_mask_static(
                input_embeds=inputs_embeds,
                inputs_embeds_shape=input_embeds_shape,
                ids_to_attend=final_ids_to_attend,
                tokens_indexing=tokens_indexing,
                device=input_device,
                dtype=input_dtype,
            )   
            mask_to_visualize = attention_mask

        if can_build_custom_mask and source_ids_to_attend is not None and has_gaze_source_indices:
            # If source_ids_to_attend is provided, we also build a mask for it.
            # This is useful for cases where we want to boost attention to specific tokens.
            source_attention_mask = LlavaQwenForCausalLM._build_custom_attention_mask_static(
                input_embeds=inputs_embeds,
                inputs_embeds_shape=input_embeds_shape,
                ids_to_attend=source_ids_to_attend,
                tokens_indexing=tokens_indexing,
                device=input_device,
                dtype=input_dtype,
                mask_all_image=False,  # Assuming we want to mask all image tokens in the source attention
            )
            # in case of no target attention mask, we fallback to source_attention_mask
            if attention_mask is None:
                attention_mask = source_attention_mask.clone()
                mask_to_visualize = attention_mask
                # source_attention_mask = None
        else:
            source_attention_mask = None

        if mask_to_visualize is None and attention_mask is not None:
            mask_to_visualize = attention_mask
        
        # attention_mask = None   # todo: remove once done testing
        # source_attention_mask = None    # todo: remove once done testing

        if inputs_embeds is None and attention_mask is not None and attention_mask.dim() > 2:
            tokens_to_take = 1
            attention_mask = attention_mask[:, :, -tokens_to_take:, :]      # reduce only to the last query token
            if source_attention_mask is not None:
                source_attention_mask = source_attention_mask[:, :, -tokens_to_take:, :]

        if attention_mask_viz_config:
            self._handle_attention_mask_viz(
                mask_to_visualize, tokens_indexing, attention_mask_viz_config
            )
        # If final_ids_to_attend is empty, attention_mask remains as is.
        # It's assumed that if images were processed, prepare_inputs_labels_for_multimodal
        # would have set up a suitable (e.g., causal) attention_mask for inputs_embeds.
        # If inputs_embeds was passed directly, then the passed attention_mask is used.
        # if kwargs.get("target_tokens", 0) == 2:
        #     tokens_to_take = 1
        #     attention_mask = attention_mask[:, :, -tokens_to_take:, :]      # reduce only to the last query token
        #     if source_attention_mask is not None:
        #         source_attention_mask = source_attention_mask[:, :, -tokens_to_take:, :]

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
            outputs = super().forward(
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
                # boost_positions=kwargs.get("boost_positions", None),
                # bias_strength=kwargs.get("bias_strength", None),
                tokens_indexing=tokens_indexing,
                # query_indices=kwargs.get("query_indices", None),
                **kwargs
            )
            if roi_requires_hidden_states:
                outputs = self._apply_roi_contrastive_loss(
                    outputs=outputs,
                    labels=labels,
                    image_features=image_features_ret,
                    roi_gaze_xy=roi_gaze_xy,
                    roi_gaze_valid=roi_gaze_valid,
                    roi_candidate_boxes=roi_candidate_boxes,
                    roi_candidate_is_positive=roi_candidate_is_positive,
                    roi_candidate_valid=roi_candidate_valid,
                )
            if forced_output_hidden_states and isinstance(outputs, CausalLMOutputWithPast):
                outputs.hidden_states = None
            return outputs

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

        image_embeds = None
        self.tokens_indexing = None
        if images is not None:
            multimodal_outputs = self.prepare_inputs_labels_for_multimodal(
            inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )

            output_len = len(multimodal_outputs)
            if output_len == 6:
                inputs, position_ids, attention_mask, _, inputs_embeds, _ = multimodal_outputs
            elif output_len >= 8:
                inputs, position_ids, attention_mask, _, inputs_embeds, _, image_embeds = multimodal_outputs[:7]
            if output_len == 9:
                self.tokens_indexing = multimodal_outputs[8]
            else:
                raise ValueError(
                    "prepare_inputs_labels_for_multimodal returned an unexpected number of values: "
                    f"{output_len}."
                )
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

    def _clone_tokens_indexing_for_viz(self, tokens_indexing: Optional[dict]) -> Optional[dict]:
        if not isinstance(tokens_indexing, dict):
            return None

        cloned: Dict[str, Any] = {}
        for key, value in tokens_indexing.items():
            if isinstance(value, list):
                cloned_list: List[torch.Tensor] = []
                for entry in value:
                    if isinstance(entry, torch.Tensor):
                        cloned_list.append(entry.detach().cpu())
                if cloned_list:
                    cloned[key] = cloned_list
            elif isinstance(value, torch.Tensor):
                cloned[key] = value.detach().cpu()
            else:
                cloned[key] = value
        return cloned

    def _handle_attention_mask_viz(
        self,
        attention_mask: Optional[torch.Tensor],
        tokens_indexing: Optional[dict],
        viz_config: Optional[Dict[str, Any]],
    ) -> None:
        self.latest_attention_mask_snapshot = None
        if attention_mask is None or not viz_config:
            return

        capture_only = bool(viz_config.get("capture_only", False))
        mask_cpu = attention_mask.detach().to(device="cpu", dtype=torch.float32)
        tokens_clone = self._clone_tokens_indexing_for_viz(tokens_indexing)

        output_dir = viz_config.get("output_dir")
        step_idx = viz_config.get("step_idx")
        token_text = viz_config.get("token_text", "<unk>")

        if (
            capture_only
            or output_dir is None
            or step_idx is None
            or viz_config.get("enabled", True) is False
        ):
            self.latest_attention_mask_snapshot = {
                "mask": mask_cpu,
                "tokens_indexing": tokens_clone,
            }
            return

        try:
            visualize_attention_mask_step(
                attention_mask=mask_cpu,
                tokens_indexing=tokens_clone,
                step_idx=int(step_idx),
                token_text=str(token_text),
                output_dir=output_dir,
                query_index=viz_config.get("query_index", None),
                system_token_indices=viz_config.get("system_token_indices", None),
                filename_prefix=viz_config.get("filename_prefix", "custom_mask"),
            )
        except Exception as exc:
            print(f"[ATTN_VIZ] Failed to visualize attention mask at step {step_idx}: {exc}")
        finally:
            self.latest_attention_mask_snapshot = None

    def pop_latest_attention_mask_snapshot(self) -> Optional[Dict[str, Any]]:
        snapshot = self.latest_attention_mask_snapshot
        self.latest_attention_mask_snapshot = None
        return snapshot

    def _ensure_roi_projection_heads(self, target_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        pos_hidden_dim = int(getattr(self.config, "roi_pos_embed_dim", 64))
        if self.roi_text_projector.out_features != target_dim:
            self.roi_text_projector = nn.Linear(self.config.hidden_size, target_dim, bias=False).to(device=device, dtype=dtype)
            self.roi_vision_projector = nn.Linear(self.config.hidden_size, target_dim, bias=False).to(device=device, dtype=dtype)
            self.roi_position_mlp = nn.Sequential(
                nn.Linear(6, pos_hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(pos_hidden_dim, target_dim, bias=False),
            ).to(device=device, dtype=dtype)
        else:
            self.roi_text_projector = self.roi_text_projector.to(device=device, dtype=dtype)
            self.roi_vision_projector = self.roi_vision_projector.to(device=device, dtype=dtype)
            self.roi_position_mlp = self.roi_position_mlp.to(device=device, dtype=dtype)

    def _compute_oof_text_projs(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not bool(getattr(self.config, "roi_contrastive_oof_enable", False)):
            return None
        token_id_sequences = getattr(self.config, "roi_contrastive_oof_token_id_sequences", None)
        if not token_id_sequences:
            fallback_ids = getattr(self.config, "roi_contrastive_oof_token_ids", None)
            if isinstance(fallback_ids, (list, tuple)) and len(fallback_ids) > 0:
                token_id_sequences = [fallback_ids]
        if not isinstance(token_id_sequences, (list, tuple)) or len(token_id_sequences) == 0:
            return None

        oof_proj_list: List[torch.Tensor] = []
        for token_ids in token_id_sequences:
            if not isinstance(token_ids, (list, tuple)) or len(token_ids) == 0:
                continue
            normalized_ids: List[int] = []
            for token_id in token_ids:
                try:
                    normalized_ids.append(int(token_id))
                except (TypeError, ValueError):
                    continue
            if not normalized_ids:
                continue

            input_ids = torch.tensor([normalized_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            oof_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if oof_outputs.hidden_states is None or len(oof_outputs.hidden_states) == 0:
                continue
            oof_hidden = oof_outputs.hidden_states[-1]
            if oof_hidden.ndim != 3 or oof_hidden.shape[1] == 0:
                continue
            oof_text_embed = oof_hidden[0].mean(dim=0)
            oof_proj = F.normalize(self.roi_text_projector(oof_text_embed.float().unsqueeze(0)), dim=-1).squeeze(0)
            oof_proj_list.append(oof_proj.to(device=device))

        if not oof_proj_list:
            return None
        return torch.stack(oof_proj_list, dim=0)

    @staticmethod
    def _build_roi_focus_span_mask(
        labels_im_sample: torch.Tensor,
        focus_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        span_mask = labels_im_sample >= 0
        if focus_start > 0:
            token_positions = torch.arange(labels_im_sample.shape[0], device=device)
            span_mask = span_mask & (token_positions >= focus_start)
        return span_mask

    @staticmethod
    def _resolve_roi_grid_side(
        sample_image_features: torch.Tensor,
        default_grid_side: int,
    ) -> Tuple[int, int]:
        grid_side = default_grid_side
        if grid_side <= 0 or grid_side * grid_side > sample_image_features.shape[0]:
            grid_side = int(math.sqrt(sample_image_features.shape[0]))
        return grid_side, grid_side * grid_side

    def _build_roi_candidate_embeddings(
        self,
        sample_image_features: torch.Tensor,
        sample_boxes: torch.Tensor,
        sample_valid: torch.Tensor,
        sample_is_positive: torch.Tensor,
        grid_side: int,
        base_token_count: int,
    ) -> Tuple[List[torch.Tensor], List[bool], List[int]]:
        candidate_embeds: List[torch.Tensor] = []
        candidate_positive_flags: List[bool] = []
        candidate_source_slots: List[int] = []

        for cand_idx in range(sample_boxes.shape[0]):
            if not bool(sample_valid[cand_idx].item()):
                continue
            x1 = float(torch.clamp(sample_boxes[cand_idx][0], 0.0, 1.0).item())
            y1 = float(torch.clamp(sample_boxes[cand_idx][1], 0.0, 1.0).item())
            x2 = float(torch.clamp(sample_boxes[cand_idx][2], 0.0, 1.0).item())
            y2 = float(torch.clamp(sample_boxes[cand_idx][3], 0.0, 1.0).item())
            bbox_patch_indices = build_bbox_patch_indices(x1, y1, x2, y2, grid_side)
            if not bbox_patch_indices:
                continue
            bbox_indices_tensor = torch.as_tensor(
                bbox_patch_indices, device=sample_image_features.device, dtype=torch.long
            )
            bbox_indices_tensor = bbox_indices_tensor[bbox_indices_tensor < base_token_count]
            if bbox_indices_tensor.numel() == 0:
                continue

            vision_embed = sample_image_features.index_select(0, bbox_indices_tensor).mean(dim=0)
            vision_proj = self.roi_vision_projector(vision_embed.float().unsqueeze(0)).squeeze(0)
            pos_vector = build_roi_position_vector(x1, y1, x2, y2)
            pos_tensor = torch.tensor(pos_vector, device=vision_proj.device, dtype=vision_proj.dtype).unsqueeze(0)
            pos_proj = self.roi_position_mlp(pos_tensor).squeeze(0)
            fused_embed = F.normalize(vision_proj + pos_proj, dim=-1)
            candidate_embeds.append(fused_embed)
            candidate_positive_flags.append(bool(sample_is_positive[cand_idx].item()))
            candidate_source_slots.append(int(cand_idx))

        return candidate_embeds, candidate_positive_flags, candidate_source_slots

    def _compute_roi_candidate_image_objective(
        self,
        candidate_tensor: torch.Tensor,
        positive_mask: torch.Tensor,
        candidate_source_slots: List[int],
        candidate_slot_count: int,
        text_proj: torch.Tensor,
        temperature: float,
        oof_enabled: bool,
        oof_text_projs: Optional[torch.Tensor],
        use_true_oof_frames: bool,
        is_true_oof: bool,
    ) -> Optional[Dict[str, Any]]:
        has_pos = bool(positive_mask.any().item())
        has_neg = bool((~positive_mask).any().item())
        cosine_scores = candidate_tensor @ text_proj
        slot_scores = torch.full(
            (candidate_slot_count,),
            float("nan"),
            device=cosine_scores.device,
            dtype=torch.float32,
        )
        for local_candidate_idx, source_slot_idx in enumerate(candidate_source_slots):
            slot_scores[int(source_slot_idx)] = torch.clamp(
                cosine_scores[local_candidate_idx], min=-1.0, max=1.0
            ).detach().float()

        # Sub-step A: standard candidate supervision (positive and negative boxes exist).
        if has_pos and has_neg:
            logits = cosine_scores / temperature
            pred_candidate_slot = -1
            if candidate_source_slots:
                pred_candidate_slot = int(candidate_source_slots[int(torch.argmax(logits).item())])
            oof_scores = None
            if oof_enabled and oof_text_projs is not None and oof_text_projs.numel() > 0:
                oof_scores = torch.clamp(candidate_tensor @ oof_text_projs.t(), min=-1.0, max=1.0).max(dim=0).values
            image_loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[positive_mask], dim=0)
            image_top1 = positive_mask[torch.argmax(logits)].float()
            return {
                "has_pos": True,
                "has_pos_and_neg": True,
                "true_oof_supervised": False,
                "pred_candidate_slot": pred_candidate_slot,
                "slot_scores": slot_scores,
                "oof_scores": oof_scores,
                "image_loss": image_loss,
                "image_top1": image_top1,
            }

        # Sub-step B: optional true-OOF supervision (OOF text as positives, candidate boxes as negatives).
        can_use_true_oof = (
            use_true_oof_frames
            and is_true_oof
            and oof_enabled
            and oof_text_projs is not None
            and oof_text_projs.numel() > 0
            and not has_pos
        )
        if not can_use_true_oof:
            return None

        oof_scores = oof_text_projs @ text_proj
        candidate_logits = cosine_scores / temperature
        oof_logits = oof_scores / temperature
        logits = torch.cat([oof_logits, candidate_logits], dim=0)
        image_loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(oof_logits, dim=0)
        top_index = int(torch.argmax(logits).item())
        image_top1 = logits.new_tensor(float(top_index < int(oof_logits.shape[0])))
        pred_candidate_slot = -1
        if top_index >= int(oof_logits.shape[0]) and candidate_source_slots:
            best_candidate_local = top_index - int(oof_logits.shape[0])
            if 0 <= best_candidate_local < len(candidate_source_slots):
                pred_candidate_slot = int(candidate_source_slots[best_candidate_local])
        return {
            "has_pos": False,
            "has_pos_and_neg": False,
            "true_oof_supervised": True,
            "pred_candidate_slot": pred_candidate_slot,
            "slot_scores": slot_scores,
            "oof_scores": torch.clamp(oof_scores, min=-1.0, max=1.0),
            "image_loss": image_loss,
            "image_top1": image_top1,
        }

    def _apply_roi_candidate_path(
        self,
        outputs: CausalLMOutputWithPast,
        labels: torch.Tensor,
        hidden_last: torch.Tensor,
        image_features: List[torch.Tensor],
        roi_candidate_boxes: torch.Tensor,
        roi_candidate_is_positive: torch.Tensor,
        roi_candidate_valid: torch.Tensor,
        roi_gaze_valid: Optional[torch.Tensor],
        phrase_sequences: List[torch.Tensor],
        default_grid_side: int,
        temperature: float,
        effective_weight: float,
        oof_enabled: bool,
        oof_weight: float,
        oof_text_projs: Optional[torch.Tensor],
        use_true_oof_frames: bool,
        preview_buffer: ROIContrastivePreviewBuffer,
        debug_this_step: bool,
        debug_steps: int,
    ) -> bool:
        # Sub-step 1: initialize per-batch bookkeeping for candidate-path supervision.
        batch_limit_candidate = min(
            hidden_last.shape[0],
            labels.shape[0],
            roi_candidate_boxes.shape[0],
            roi_candidate_is_positive.shape[0],
            roi_candidate_valid.shape[0],
            len(image_features),
        )
        if roi_gaze_valid is not None:
            batch_limit_candidate = min(batch_limit_candidate, roi_gaze_valid.shape[0])
        image_losses: List[torch.Tensor] = []
        image_top1: List[torch.Tensor] = []
        candidate_count_total = 0
        candidate_text_bank: List[torch.Tensor] = []
        candidate_oof_anchors: List[torch.Tensor] = []
        candidate_oof_targets: List[int] = []
        samples_focus_found = 0
        samples_span_valid = 0
        samples_image_valid = 0
        samples_with_candidates = 0
        samples_with_pos = 0
        samples_with_pos_and_neg = 0
        samples_true_oof_supervised = 0

        # Sub-step 2: build image-level candidate/text representations and image-level objectives.
        for im_idx in range(batch_limit_candidate):
            focus_start = locate_focus_start_index(labels[im_idx], phrase_sequences)
            if focus_start is None:
                continue
            samples_focus_found += 1
            span_mask = self._build_roi_focus_span_mask(labels[im_idx], focus_start, labels.device)
            if not span_mask.any():
                continue
            samples_span_valid += 1
            text_embed = hidden_last[im_idx][span_mask].mean(dim=0)

            sample_image_features = image_features[im_idx]
            if not torch.is_tensor(sample_image_features) or sample_image_features.ndim != 2:
                continue
            samples_image_valid += 1
            grid_side, base_token_count = self._resolve_roi_grid_side(sample_image_features, default_grid_side)
            if base_token_count <= 0:
                continue

            sample_boxes = roi_candidate_boxes[im_idx]
            sample_is_positive = roi_candidate_is_positive[im_idx].bool()
            sample_valid = roi_candidate_valid[im_idx].bool()
            candidate_embeds, candidate_positive_flags, candidate_source_slots = self._build_roi_candidate_embeddings(
                sample_image_features=sample_image_features,
                sample_boxes=sample_boxes,
                sample_valid=sample_valid,
                sample_is_positive=sample_is_positive,
                grid_side=grid_side,
                base_token_count=base_token_count,
            )

            if candidate_embeds:
                samples_with_candidates += 1
            if not candidate_embeds:
                continue

            positive_mask = torch.tensor(candidate_positive_flags, device=hidden_last.device, dtype=torch.bool)
            has_pos = bool(positive_mask.any().item())
            has_neg = bool((~positive_mask).any().item())
            if has_pos:
                samples_with_pos += 1
            if has_pos and has_neg:
                samples_with_pos_and_neg += 1

            candidate_tensor = torch.stack(candidate_embeds, dim=0)
            text_proj = F.normalize(self.roi_text_projector(text_embed.float().unsqueeze(0)), dim=-1).squeeze(0)
            image_obj = self._compute_roi_candidate_image_objective(
                candidate_tensor=candidate_tensor,
                positive_mask=positive_mask,
                candidate_source_slots=candidate_source_slots,
                candidate_slot_count=int(sample_boxes.shape[0]),
                text_proj=text_proj,
                temperature=temperature,
                oof_enabled=oof_enabled,
                oof_text_projs=oof_text_projs,
                use_true_oof_frames=use_true_oof_frames,
                is_true_oof=(roi_gaze_valid is not None and not bool(roi_gaze_valid[im_idx].item())),
            )
            if image_obj is None:
                continue

            if bool(image_obj["true_oof_supervised"]):
                samples_true_oof_supervised += 1
            preview_buffer.append(
                im_idx,
                int(image_obj["pred_candidate_slot"]),
                image_obj["slot_scores"],
                oof_scores=image_obj["oof_scores"],
            )
            image_losses.append(image_obj["image_loss"])
            image_top1.append(image_obj["image_top1"])
            candidate_count_total += int(candidate_tensor.shape[0])
            candidate_text_bank.append(text_proj)
            text_index = len(candidate_text_bank) - 1
            if bool(image_obj["has_pos"]) and oof_enabled and roi_gaze_valid is not None and bool(roi_gaze_valid[im_idx].item()):
                positive_anchor = F.normalize(candidate_tensor[positive_mask].mean(dim=0), dim=-1)
                candidate_oof_anchors.append(positive_anchor)
                candidate_oof_targets.append(text_index)

        # Sub-step 3: aggregate image losses, optionally add OOF-anchor term, and write final stats.
        if image_losses or (oof_enabled and len(candidate_oof_anchors) > 0):
            stats_device = outputs.loss.device
            nce_loss = (
                torch.stack(image_losses).mean()
                if image_losses
                else torch.zeros((), device=hidden_last.device, dtype=torch.float32)
            )
            top1_acc = (
                torch.stack(image_top1).mean()
                if image_top1
                else torch.zeros((), device=hidden_last.device, dtype=torch.float32)
            )
            total_roi_loss = nce_loss
            oof_loss = torch.zeros((), device=hidden_last.device, dtype=torch.float32)
            top1_with_oof = top1_acc
            oof_sample_count = torch.zeros((), device=hidden_last.device, dtype=torch.long)
            if oof_enabled and len(candidate_oof_anchors) > 0 and oof_text_projs is not None:
                text_bank = torch.cat([torch.stack(candidate_text_bank, dim=0), oof_text_projs], dim=0)
                anchor_batch = torch.stack(candidate_oof_anchors, dim=0)
                targets_oof = torch.tensor(candidate_oof_targets, device=hidden_last.device, dtype=torch.long)
                oof_logits = (anchor_batch @ text_bank.t()) / temperature
                oof_loss = F.cross_entropy(oof_logits, targets_oof)
                top1_with_oof = (oof_logits.argmax(dim=-1) == targets_oof).float().mean()
                oof_sample_count = torch.tensor(len(candidate_oof_targets), device=hidden_last.device, dtype=torch.long)
                total_roi_loss = total_roi_loss + total_roi_loss.new_tensor(oof_weight) * oof_loss

            outputs.loss = outputs.loss + outputs.loss.new_tensor(effective_weight) * total_roi_loss.to(
                device=outputs.loss.device,
                dtype=outputs.loss.dtype,
            )
            (
                preview_image_index_tensor,
                preview_pred_slot_tensor,
                preview_candidate_slot_score_tensor,
                preview_oof_score_tensor,
            ) = preview_buffer.to_tensors(stats_device)
            self._roi_contrastive_stats = {
                "nce_loss": nce_loss.detach(),
                "top1": top1_acc.detach(),
                "top1_with_oof": top1_with_oof.detach(),
                "pairs": torch.tensor(candidate_count_total, device=stats_device, dtype=torch.long),
                "lambda": torch.tensor(effective_weight, device=stats_device, dtype=torch.float32),
                "preview_image_indices": preview_image_index_tensor,
                "preview_pred_candidate_slots": preview_pred_slot_tensor,
                "preview_candidate_slot_scores": preview_candidate_slot_score_tensor,
                "preview_oof_scores": preview_oof_score_tensor,
            }
            if oof_enabled:
                self._roi_contrastive_stats["oof_loss"] = oof_loss.detach().to(device=stats_device, dtype=torch.float32)
                self._roi_contrastive_stats["oof_samples"] = oof_sample_count.detach().to(device=stats_device, dtype=torch.long)
            if debug_this_step:
                roi_debug_log(
                    step=self._roi_contrastive_step,
                    debug_steps=debug_steps,
                    message=(
                        f"candidate_path: samples_total={batch_limit_candidate}, focus={samples_focus_found}, span={samples_span_valid}, "
                        f"images_valid={samples_image_valid}, has_cands={samples_with_candidates}, has_pos={samples_with_pos}, "
                        f"has_pos_neg={samples_with_pos_and_neg}, true_oof_samples={samples_true_oof_supervised}, "
                        f"image_losses={len(image_losses)}, pairs={candidate_count_total}, "
                        f"oof_samples={int(oof_sample_count.item())}, nce={float(nce_loss.item()):.6f}, oof={float(oof_loss.item()):.6f}"
                    ),
                )
            return True

        if debug_this_step:
            roi_debug_log(
                step=self._roi_contrastive_step,
                debug_steps=debug_steps,
                message=(
                    f"candidate_path_skipped: samples_total={batch_limit_candidate}, focus={samples_focus_found}, span={samples_span_valid}, "
                    f"images_valid={samples_image_valid}, has_cands={samples_with_candidates}, has_pos={samples_with_pos}, "
                    f"has_pos_neg={samples_with_pos_and_neg}, true_oof_samples={samples_true_oof_supervised}, "
                    f"image_losses={len(image_losses)}"
                ),
            )
        return False

    def _apply_roi_contrastive_loss(
        self,
        outputs: CausalLMOutputWithPast,
        labels: Optional[torch.Tensor],
        image_features: Optional[List[torch.Tensor]],
        roi_gaze_xy: Optional[torch.Tensor],
        roi_gaze_valid: Optional[torch.Tensor],
        roi_candidate_boxes: Optional[torch.Tensor] = None,
        roi_candidate_is_positive: Optional[torch.Tensor] = None,
        roi_candidate_valid: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Add ROI contrastive supervision on top of the language-model loss.

        Execution flow:
        1. Validate inputs and initialize ROI projection/scheduling state.
        2. Prefer candidate-box supervision when ROI candidate metadata exists.
           Optionally, true-OOF rows can use OOF text anchors as positives.
        3. Fall back to gaze-centered circular ROI supervision otherwise.
        4. Store debug/preview tensors in `self._roi_contrastive_stats`.
        """
        if not isinstance(outputs, CausalLMOutputWithPast):
            return outputs
        if outputs.loss is None or outputs.hidden_states is None or labels is None:
            return outputs
        if image_features is None:
            return outputs

        self._roi_contrastive_step += 1
        phrase_sequences = prepare_focus_phrase_sequences(
            getattr(self.config, "roi_contrastive_phrase_token_ids", None),
            labels.device,
        )
        if not phrase_sequences:
            return outputs

        hidden_last = outputs.hidden_states[-1]
        batch_limit = min(hidden_last.shape[0], labels.shape[0], len(image_features))
        if roi_gaze_xy is not None:
            batch_limit = min(batch_limit, roi_gaze_xy.shape[0])
        if roi_gaze_valid is not None:
            batch_limit = min(batch_limit, roi_gaze_valid.shape[0])
        if batch_limit <= 0:
            return outputs

        # Step 1: collect runtime knobs and ensure projection heads match configured ROI dim.
        vision_tower = self.get_vision_tower()
        default_grid_side = 0
        if vision_tower is not None:
            default_grid_side = int(getattr(vision_tower, "num_patches_per_side", 0) or 0)
        temperature = max(float(getattr(self.config, "roi_contrastive_temperature", 0.07)), 1e-6)
        shared_dim = int(getattr(self.config, "roi_contrastive_dim", 512))
        self._ensure_roi_projection_heads(shared_dim, device=hidden_last.device, dtype=hidden_last.dtype)
        base_weight = float(getattr(self.config, "roi_contrastive_weight", 0.1))
        warmup_steps = int(getattr(self.config, "roi_contrastive_warmup_steps", 0))
        warmup_factor = 1.0
        if warmup_steps > 0:
            warmup_factor = min(1.0, float(self._roi_contrastive_step) / float(warmup_steps))
        effective_weight = base_weight * warmup_factor
        debug_steps = int(getattr(self.config, "roi_contrastive_debug_steps", 20))
        debug_this_step = debug_steps > 0 and self._roi_contrastive_step <= debug_steps
        oof_enabled = bool(getattr(self.config, "roi_contrastive_oof_enable", False))
        oof_weight = max(0.0, float(getattr(self.config, "roi_contrastive_oof_weight", 0.5)))
        use_true_oof_frames = bool(getattr(self.config, "roi_contrastive_use_true_oof_frames", False))
        preview_samples = max(1, int(getattr(self.config, "roi_contrastive_preview_samples", 5)))
        oof_text_projs = self._compute_oof_text_projs(device=hidden_last.device, dtype=hidden_last.dtype) if oof_enabled else None
        if oof_enabled and oof_text_projs is None:
            oof_enabled = False

        preview_buffer = ROIContrastivePreviewBuffer(preview_samples)

        candidate_inputs_ready = (
            roi_candidate_boxes is not None
            and roi_candidate_is_positive is not None
            and roi_candidate_valid is not None
            and roi_candidate_boxes.ndim == 3
            and roi_candidate_is_positive.ndim == 2
            and roi_candidate_valid.ndim == 2
        )
        if debug_this_step:
            valid_gaze_count = -1
            if roi_gaze_valid is not None:
                valid_gaze_count = int(roi_gaze_valid[:batch_limit].bool().sum().item())
            roi_debug_log(
                step=self._roi_contrastive_step,
                debug_steps=debug_steps,
                message=(
                    f"batch_limit={batch_limit}, candidate_inputs_ready={candidate_inputs_ready}, "
                    f"roi_gaze_valid_count={valid_gaze_count}, "
                    f"warmup_steps={warmup_steps}, warmup_factor={warmup_factor:.4f}, "
                    f"base_weight={base_weight:.4f}, effective_weight={effective_weight:.6f}, "
                    f"use_true_oof_frames={use_true_oof_frames}"
                ),
            )
        # Step 2: candidate-box path (preferred when explicit positive/negative ROIs exist).
        if candidate_inputs_ready:
            candidate_applied = self._apply_roi_candidate_path(
                outputs=outputs,
                labels=labels,
                hidden_last=hidden_last,
                image_features=image_features,
                roi_candidate_boxes=roi_candidate_boxes,
                roi_candidate_is_positive=roi_candidate_is_positive,
                roi_candidate_valid=roi_candidate_valid,
                roi_gaze_valid=roi_gaze_valid,
                phrase_sequences=phrase_sequences,
                default_grid_side=default_grid_side,
                temperature=temperature,
                effective_weight=effective_weight,
                oof_enabled=oof_enabled,
                oof_weight=oof_weight,
                oof_text_projs=oof_text_projs,
                use_true_oof_frames=use_true_oof_frames,
                preview_buffer=preview_buffer,
                debug_this_step=debug_this_step,
                debug_steps=debug_steps,
            )
            if candidate_applied:
                return outputs

        # Step 3: gaze-centered fallback path when candidate metadata is missing or unusable.
        if roi_gaze_xy is None or roi_gaze_valid is None:
            return outputs

        radius_ratio = float(getattr(self.config, "roi_contrastive_radius_ratio", 0.08))
        text_embeddings: List[torch.Tensor] = []
        vision_embeddings: List[torch.Tensor] = []

        for im_idx in range(batch_limit):
            if not bool(roi_gaze_valid[im_idx].item()):
                continue

            focus_start = locate_focus_start_index(labels[im_idx], phrase_sequences)
            if focus_start is None:
                continue
            span_mask = self._build_roi_focus_span_mask(labels[im_idx], focus_start, labels.device)
            if not span_mask.any():
                continue
            text_embed = hidden_last[im_idx][span_mask].mean(dim=0)

            sample_image_features = image_features[im_idx]
            if not torch.is_tensor(sample_image_features) or sample_image_features.ndim != 2:
                continue
            grid_side, base_token_count = self._resolve_roi_grid_side(sample_image_features, default_grid_side)
            if base_token_count <= 0:
                continue

            x_norm = float(torch.clamp(roi_gaze_xy[im_idx][0], 0.0, 1.0).item())
            y_norm = float(torch.clamp(roi_gaze_xy[im_idx][1], 0.0, 1.0).item())
            radius = max(1, int(round(radius_ratio * grid_side)))
            roi_indices = build_circular_roi_patch_indices(x_norm, y_norm, grid_side, radius)
            if not roi_indices:
                continue
            roi_indices_tensor = torch.as_tensor(roi_indices, device=sample_image_features.device, dtype=torch.long)
            roi_indices_tensor = roi_indices_tensor[roi_indices_tensor < base_token_count]
            if roi_indices_tensor.numel() == 0:
                continue
            vision_embed = sample_image_features.index_select(0, roi_indices_tensor).mean(dim=0)

            text_embeddings.append(text_embed)
            vision_embeddings.append(vision_embed)

        pair_count = len(text_embeddings)
        stats_device = outputs.loss.device
        if pair_count < 1:
            (
                preview_image_index_tensor,
                preview_pred_slot_tensor,
                preview_candidate_slot_score_tensor,
                preview_oof_score_tensor,
            ) = preview_buffer.to_tensors(stats_device)
            self._roi_contrastive_stats = {
                "nce_loss": torch.zeros((), device=stats_device, dtype=torch.float32),
                "top1": torch.zeros((), device=stats_device, dtype=torch.float32),
                "top1_with_oof": torch.zeros((), device=stats_device, dtype=torch.float32),
                "pairs": torch.tensor(pair_count, device=stats_device, dtype=torch.long),
                "lambda": torch.zeros((), device=stats_device, dtype=torch.float32),
                "preview_image_indices": preview_image_index_tensor,
                "preview_pred_candidate_slots": preview_pred_slot_tensor,
                "preview_candidate_slot_scores": preview_candidate_slot_score_tensor,
                "preview_oof_scores": preview_oof_score_tensor,
            }
            if oof_enabled:
                self._roi_contrastive_stats["oof_loss"] = torch.zeros((), device=stats_device, dtype=torch.float32)
                self._roi_contrastive_stats["oof_samples"] = torch.zeros((), device=stats_device, dtype=torch.long)
            return outputs

        if pair_count < 2 and not self._roi_contrastive_warned_small_pair_count:
            print(
                "[ROI contrastive] pair_count < 2 in current micro-batch; "
                "InfoNCE is skipped and roi_contrastive/nce_loss may stay near zero. "
                "Enable OOF negatives or increase per-device train batch size (>=2)."
            )
            self._roi_contrastive_warned_small_pair_count = True

        text_batch = torch.stack(text_embeddings, dim=0)
        vision_batch = torch.stack(vision_embeddings, dim=0)
        text_proj = F.normalize(self.roi_text_projector(text_batch.float()), dim=-1)
        vision_proj = F.normalize(self.roi_vision_projector(vision_batch.float()), dim=-1)
        similarity = (vision_proj @ text_proj.t()) / temperature
        nce_loss = torch.zeros((), device=hidden_last.device, dtype=torch.float32)
        top1_acc = torch.zeros((), device=hidden_last.device, dtype=torch.float32)
        if pair_count >= 2:
            targets = torch.arange(pair_count, device=similarity.device, dtype=torch.long)
            loss_v2t = F.cross_entropy(similarity, targets)
            loss_t2v = F.cross_entropy(similarity.t(), targets)
            nce_loss = 0.5 * (loss_v2t + loss_t2v)
            top1_acc = (similarity.argmax(dim=-1) == targets).float().mean()

        total_roi_loss = nce_loss
        oof_loss = torch.zeros((), device=hidden_last.device, dtype=torch.float32)
        top1_with_oof = top1_acc
        oof_sample_count = torch.zeros((), device=hidden_last.device, dtype=torch.long)
        has_oof_term = False
        if oof_enabled and oof_text_projs is not None:
            text_bank = torch.cat([text_proj, oof_text_projs], dim=0)
            oof_logits = (vision_proj @ text_bank.t()) / temperature
            oof_targets = torch.arange(pair_count, device=oof_logits.device, dtype=torch.long)
            oof_loss = F.cross_entropy(oof_logits, oof_targets)
            top1_with_oof = (oof_logits.argmax(dim=-1) == oof_targets).float().mean()
            oof_sample_count = torch.tensor(pair_count, device=hidden_last.device, dtype=torch.long)
            total_roi_loss = total_roi_loss + total_roi_loss.new_tensor(oof_weight) * oof_loss
            has_oof_term = True

        has_base_term = pair_count >= 2
        lambda_value = effective_weight if (has_base_term or has_oof_term) else 0.0
        if has_base_term or has_oof_term:
            outputs.loss = outputs.loss + outputs.loss.new_tensor(effective_weight) * total_roi_loss.to(
                device=outputs.loss.device,
                dtype=outputs.loss.dtype,
            )

        (
            preview_image_index_tensor,
            preview_pred_slot_tensor,
            preview_candidate_slot_score_tensor,
            preview_oof_score_tensor,
        ) = preview_buffer.to_tensors(stats_device)
        self._roi_contrastive_stats = {
            "nce_loss": nce_loss.detach(),
            "top1": top1_acc.detach(),
            "top1_with_oof": top1_with_oof.detach(),
            "pairs": torch.tensor(pair_count, device=stats_device, dtype=torch.long),
            "lambda": torch.tensor(lambda_value, device=stats_device, dtype=torch.float32),
            "preview_image_indices": preview_image_index_tensor,
            "preview_pred_candidate_slots": preview_pred_slot_tensor,
            "preview_candidate_slot_scores": preview_candidate_slot_score_tensor,
            "preview_oof_scores": preview_oof_score_tensor,
        }
        if oof_enabled:
            self._roi_contrastive_stats["oof_loss"] = oof_loss.detach().to(device=stats_device, dtype=torch.float32)
            self._roi_contrastive_stats["oof_samples"] = oof_sample_count.detach().to(device=stats_device, dtype=torch.long)
        if debug_this_step:
            roi_debug_log(
                step=self._roi_contrastive_step,
                debug_steps=debug_steps,
                message=(
                    f"fallback_path: pair_count={pair_count}, has_base_term={has_base_term}, has_oof_term={has_oof_term}, "
                    f"nce={float(nce_loss.item()):.6f}, oof={float(oof_loss.item()):.6f}, "
                    f"lambda={float(lambda_value):.6f}"
                ),
            )
        return outputs

    def pop_roi_contrastive_stats(self) -> Optional[Dict[str, torch.Tensor]]:
        stats = self._roi_contrastive_stats
        self._roi_contrastive_stats = None
        return stats

    @staticmethod
    def _build_custom_attention_mask_static(input_embeds: Optional[torch.Tensor],
                                            inputs_embeds_shape: Tuple[int, ...],
                                            ids_to_attend: List[int],
                                            tokens_indexing: dict,
                                            device: torch.device,
                                            dtype: torch.dtype = torch.float16,
                                            mask_all_image=False,
                                            mask_all_text=False) -> torch.Tensor:
        """
        Builds a custom attention mask.
        The mask allows causal attention for all tokens.
        For the last token, it ONLY allows attention to `ids_to_attend`.
        """
        batch_size, seq_len, key_len = inputs_embeds_shape

        # Initialize with causal mask properties
        mask = torch.full((seq_len, seq_len), float("0"), device=device, dtype=dtype)
        causal_indices = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        mask[causal_indices] = 1.
        image_token_indices = []
        text_token_indices = []
        if isinstance(tokens_indexing, dict):
            image_list = tokens_indexing.get("image")
            text_indices = tokens_indexing.get("text")
            if (
                isinstance(image_list, list)
                and len(image_list) > 0
                and isinstance(image_list[0], torch.Tensor)
                and image_list[0].numel() > 0
            ):
                image_token_indices = image_list[0].to(device=device, dtype=torch.long)

        if mask_all_image and len(image_token_indices) > 0:
            mask[:, image_token_indices] = 0.

        if input_embeds is None and len(image_token_indices) > 0:
             # lets mask all the image tokens
            mask[:, image_token_indices] = 0.
            # mask[image_token_indices, :] = 0.

        # Modify the last query position for specific attention if ids_to_attend is provided
        if ids_to_attend and seq_len > 0:
            last_token_idx = seq_len - 1
            # mask[last_token_idx, :] = float("0")  # Disallow all by default for the last token

            valid_ids_to_attend = [idx for idx in ids_to_attend if 0 <= idx < seq_len]
            valid_ids_to_attend = torch.tensor(valid_ids_to_attend, device=device, dtype=torch.long)
            if any(valid_ids_to_attend):
                mask[last_token_idx, valid_ids_to_attend] = 1.
            # if mask_all_image:
            #     mask[:, valid_ids_to_attend] = 1.
        if not mask_all_text:
            text_token_indices = text_indices[0].to(device=device, dtype=torch.long)
            mask[last_token_idx, text_token_indices] = 1.
        # Reshape to [batch_size, 1, seq_len, seq_len] for broadcasting with attention heads
        # Qwen2 expects (batch_size, num_heads, query_length, kv_length) or (batch_size, 1, query_length, kv_length)
        # The original code produced [1,1,SL,SL]. We assume batch_size is handled by broadcasting if this is [1,1,SL,SL]
        # or we make it [B,1,SL,SL]
        mask = mask.unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        if key_len == 1:
            mask = mask[:, :, -1:, :]  # (B, 1, SL, 1)
        return mask


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
