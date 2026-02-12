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
        self.roi_text_projector = nn.Linear(config.hidden_size, roi_dim, bias=False)
        self.roi_vision_projector = nn.Linear(config.hidden_size, roi_dim, bias=False)
        self._roi_contrastive_step = 0
        self._roi_contrastive_stats: Optional[Dict[str, torch.Tensor]] = None
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
        for row in reference_input_ids:
            image_inds = (row == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
            text_inds = (row != IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
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
        roi_contrastive_enabled = bool(getattr(self.config, "roi_contrastive_enable", False))
        roi_requires_hidden_states = (
            roi_contrastive_enabled
            and self.training
            and not dpo_forward
            and images is not None
            and labels is not None
            and roi_gaze_xy is not None
            and roi_gaze_valid is not None
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
        if self.roi_text_projector.out_features != target_dim:
            self.roi_text_projector = nn.Linear(self.config.hidden_size, target_dim, bias=False).to(device=device, dtype=dtype)
            self.roi_vision_projector = nn.Linear(self.config.hidden_size, target_dim, bias=False).to(device=device, dtype=dtype)
        else:
            self.roi_text_projector = self.roi_text_projector.to(device=device, dtype=dtype)
            self.roi_vision_projector = self.roi_vision_projector.to(device=device, dtype=dtype)

    @staticmethod
    def _build_roi_patch_indices(x_norm: float, y_norm: float, grid_side: int, radius: int) -> List[int]:
        center_x = int(round(x_norm * (grid_side - 1)))
        center_y = int(round(y_norm * (grid_side - 1)))
        valid_indices: List[int] = []
        r2 = radius * radius
        min_x = max(0, center_x - radius)
        max_x = min(grid_side - 1, center_x + radius)
        min_y = max(0, center_y - radius)
        max_y = min(grid_side - 1, center_y + radius)
        for y_coord in range(min_y, max_y + 1):
            dy = y_coord - center_y
            for x_coord in range(min_x, max_x + 1):
                dx = x_coord - center_x
                if dx * dx + dy * dy <= r2:
                    valid_indices.append(y_coord * grid_side + x_coord)
        return valid_indices

    def _apply_roi_contrastive_loss(
        self,
        outputs: CausalLMOutputWithPast,
        labels: Optional[torch.Tensor],
        image_features: Optional[List[torch.Tensor]],
        roi_gaze_xy: Optional[torch.Tensor],
        roi_gaze_valid: Optional[torch.Tensor],
    ) -> CausalLMOutputWithPast:
        if not isinstance(outputs, CausalLMOutputWithPast):
            return outputs
        if outputs.loss is None or outputs.hidden_states is None or labels is None:
            return outputs
        if image_features is None or roi_gaze_xy is None or roi_gaze_valid is None:
            return outputs

        self._roi_contrastive_step += 1
        phrase_sequences = prepare_focus_phrase_sequences(
            getattr(self.config, "roi_contrastive_phrase_token_ids", None),
            labels.device,
        )
        if not phrase_sequences:
            return outputs

        hidden_last = outputs.hidden_states[-1]
        batch_limit = min(
            hidden_last.shape[0],
            labels.shape[0],
            roi_gaze_xy.shape[0],
            roi_gaze_valid.shape[0],
            len(image_features),
        )
        if batch_limit <= 0:
            return outputs

        vision_tower = self.get_vision_tower()
        default_grid_side = 0
        if vision_tower is not None:
            default_grid_side = int(getattr(vision_tower, "num_patches_per_side", 0) or 0)

        radius_ratio = float(getattr(self.config, "roi_contrastive_radius_ratio", 0.08))
        text_embeddings: List[torch.Tensor] = []
        vision_embeddings: List[torch.Tensor] = []

        for row_idx in range(batch_limit):
            if not bool(roi_gaze_valid[row_idx].item()):
                continue

            focus_start = locate_focus_start_index(labels[row_idx], phrase_sequences)
            if focus_start is None:
                continue
            span_mask = labels[row_idx] >= 0
            if focus_start > 0:
                span_mask = span_mask & (torch.arange(labels[row_idx].shape[0], device=labels.device) >= focus_start)
            if not span_mask.any():
                continue
            text_embed = hidden_last[row_idx][span_mask].mean(dim=0)

            sample_image_features = image_features[row_idx]
            if not torch.is_tensor(sample_image_features) or sample_image_features.ndim != 2:
                continue
            grid_side = default_grid_side
            if grid_side <= 0 or grid_side * grid_side > sample_image_features.shape[0]:
                grid_side = int(math.sqrt(sample_image_features.shape[0]))
            base_token_count = grid_side * grid_side
            if base_token_count <= 0:
                continue

            x_norm = float(torch.clamp(roi_gaze_xy[row_idx][0], 0.0, 1.0).item())
            y_norm = float(torch.clamp(roi_gaze_xy[row_idx][1], 0.0, 1.0).item())
            radius = max(1, int(round(radius_ratio * grid_side)))
            roi_indices = self._build_roi_patch_indices(x_norm, y_norm, grid_side, radius)
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
        if pair_count < 2:
            stats_device = outputs.loss.device
            self._roi_contrastive_stats = {
                "nce_loss": torch.zeros((), device=stats_device, dtype=torch.float32),
                "top1": torch.zeros((), device=stats_device, dtype=torch.float32),
                "pairs": torch.tensor(pair_count, device=stats_device, dtype=torch.long),
                "lambda": torch.zeros((), device=stats_device, dtype=torch.float32),
            }
            return outputs

        text_batch = torch.stack(text_embeddings, dim=0)
        vision_batch = torch.stack(vision_embeddings, dim=0)
        shared_dim = int(getattr(self.config, "roi_contrastive_dim", 512))
        self._ensure_roi_projection_heads(shared_dim, device=text_batch.device, dtype=text_batch.dtype)
        text_proj = F.normalize(self.roi_text_projector(text_batch.float()), dim=-1)
        vision_proj = F.normalize(self.roi_vision_projector(vision_batch.float()), dim=-1)

        temperature = max(float(getattr(self.config, "roi_contrastive_temperature", 0.07)), 1e-6)
        similarity = (vision_proj @ text_proj.t()) / temperature
        targets = torch.arange(pair_count, device=similarity.device, dtype=torch.long)
        loss_v2t = F.cross_entropy(similarity, targets)
        loss_t2v = F.cross_entropy(similarity.t(), targets)
        nce_loss = 0.5 * (loss_v2t + loss_t2v)

        base_weight = float(getattr(self.config, "roi_contrastive_weight", 0.1))
        warmup_steps = int(getattr(self.config, "roi_contrastive_warmup_steps", 0))
        warmup_factor = 1.0
        if warmup_steps > 0:
            warmup_factor = min(1.0, float(self._roi_contrastive_step) / float(warmup_steps))
        effective_weight = base_weight * warmup_factor

        outputs.loss = outputs.loss + outputs.loss.new_tensor(effective_weight) * nce_loss.to(
            device=outputs.loss.device,
            dtype=outputs.loss.dtype,
        )
        top1_acc = (similarity.argmax(dim=-1) == targets).float().mean()
        self._roi_contrastive_stats = {
            "nce_loss": nce_loss.detach(),
            "top1": top1_acc.detach(),
            "pairs": torch.tensor(pair_count, device=top1_acc.device, dtype=torch.long),
            "lambda": torch.tensor(effective_weight, device=top1_acc.device, dtype=torch.float32),
        }
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

        # Modify the last row for specific attention if ids_to_attend is provided
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
