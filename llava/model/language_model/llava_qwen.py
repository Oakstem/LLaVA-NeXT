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
from torch.nn import CrossEntropyLoss
import ast
import re
import math
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config        #, Qwen2Model, Qwen2ForCausalLM
from qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
from llava.mm_utils import select_best_resolution # Import the helper from mm_utils
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.attention_mask_visualizer import visualize_attention_mask_step

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
            if special_inds.numel() > 2:
                tokens_indexing["insert_embd"] = {
                    "source": [special_inds[0].item(), special_inds[2].item()],
                    "target": [special_inds[1].item()],
                }
            elif special_inds.numel() == 2:
                tokens_indexing["insert_embd"] = {
                    "source": [special_inds[0].item()],
                    "target": [special_inds[1].item()],
                }
            elif special_inds.numel() == 1:
                tokens_indexing["insert_embd"] = {"source": [special_inds[0].item()], "target": []}
        else:
            tokens_indexing["insert_embd"] = None

        return tokens_indexing

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
        self.latest_person_mask_repr = None
        ids_to_attend_pixels = []
        original_input_ids = input_ids # Save before potential modification
        # original_attention_mask = attention_mask # Keep a reference if needed
        final_ids_to_attend = kwargs.get("boost_positions" , None)
        mask_logic_applied = False
        image_token_filter_indices = kwargs.pop("image_token_filter_indices", None)
        attention_mask_viz_config = kwargs.pop("attention_mask_viz", None)

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
            except Exception as exc:
                print(f"Warning: failed to build tokens_indexing from prompt tokens: {exc}")
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
                # boost_positions=kwargs.get("boost_positions", None),
                # bias_strength=kwargs.get("bias_strength", None),
                tokens_indexing=tokens_indexing,
                # query_indices=kwargs.get("query_indices", None),
                **kwargs
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
