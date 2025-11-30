import math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict
from llava.mm_utils import select_best_resolution  # Import the helper from mm_utils

def prepare_gaze_follow_dataset(annot_path: str, data_base_dir: str):
    """
    Prepare the gaze follow dataset from the annotations file and the data base directory.
    """
    df = pd.read_csv(annot_path, sep="\t", header=None)
    # split the columns with ',' delimeter
    df = df[0].str.split(",", expand=True)
    # add the columns names:
    # [image_path,id,body_bbox_x,body_bbox_y,body_bbox_width,body_bbox_height,eye_x,eye_y,gaze_x,gaze_y,head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,in_or_out,meta]
    if len(df.columns) == 17:
        df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                            'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min',
                            'head_bbox_x_max', 'head_bbox_y_max', 'in_or_out', 'meta', 'original_path']
    else:
        df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
            'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'head_bbox_x_min', 'head_bbox_y_min',
            'head_bbox_x_max', 'head_bbox_y_max', 'meta', 'original_path']

    # to numeric
    # Convert all the numerical columns to numeric types
    numeric_columns = ['id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                    'eye_x', 'eye_y', 'gaze_x', 'gaze_y',
                    'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # After converting, you may want to check for NaN values that resulted from conversion errors
    nan_counts = df[numeric_columns].isna().sum()
    if nan_counts.sum() > 0:
        print("NaN counts after conversion:")
        print(nan_counts[nan_counts > 0])  # Only show columns that have NaNs
    # Since every image has several annotations (mostly only in test set), we need to group the annotations by image and average the gaze points
    # group by image_path
    compact_df = df.groupby('image_path').agg({
        'eye_x': 'mean',
        'eye_y': 'mean',
        'gaze_x': 'mean',
        'gaze_y': 'mean',
        'body_bbox_x': 'mean',
        'body_bbox_y': 'mean',
        'body_bbox_width': 'mean',
        'body_bbox_height': 'mean',
    }).reset_index()
    results_dict = {
                'df': df,
                'compact_df': compact_df
    }
    return results_dict


# Helper function to convert pixel coordinates to token indices
def _pixel_to_token_indices_helper_anyres(
    pixel_coords: List[Tuple[int, int]],  # List of (y, x) coordinates from top-left of original image
    original_image_size: Tuple[int, int],  # (original_height, original_width)
    possible_resolutions: List[Tuple[int, int]],  # List of (W,H) tuples from grid_pinpoints
    final_patch_division_size: int=384,  # The size used in `divide_to_patches`, e.g., processor.crop_size["height"]
    image_token_start_index_in_embeds: int=14,
    patch_size: int=14,
    add_system_prompt_tokens: bool=False,
    add_user_prompt_tokens: bool=False,
    user_prompt_len: Optional[int] = 20,
    system_prompt_len: Optional[int] = 14,
    user_prompt_range: Optional[Union[int, Tuple[int, int]]] = [1849, 1869],
    bias_offset: int = 0     
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Converts pixel coordinates to token indices for 'anyres' processed images.
    Accounts for the global image patch and subsequent patches from the padded image.
    """
    """ todo:
     1. for every coordinate, find the base image patch indices, and the anyres based patch indices
     2. check if during flatting every anyres [large] patch is being flattened separately, or they all flattented as one
     3. to find anyres based patch indices, we need to account for: - newline patch at the end of every row, - unpadded structure
    """
    original_w, original_h = original_image_size  # Note: mm_utils.select_best_resolution expects (width, height)

    # 1. Determine the "best_resolution" the image was padded/resized to.
    # select_best_resolution expects (width, height)
    best_processing_resolution_w, best_processing_resolution_h = select_best_resolution(
        (original_w, original_h), possible_resolutions
    )

    # Determine scaling factor based on which dimension to fill in best_processing_resolution
    scale_w_factor = best_processing_resolution_w / original_w
    scale_h_factor = best_processing_resolution_h / original_h

    token_indices = []
    base_xy_coords = []
    resized_mask = np.zeros([final_patch_division_size//patch_size, final_patch_division_size//patch_size], dtype=np.float32)
    for y_pixel_orig, x_pixel_orig in pixel_coords:     # coordinates are inverted: row, col
        # 2. Transform pixel coordinates from original image to the padded "best_processing_resolution".
        # This simulates resize_and_pad_image logic.
        # resize_and_pad_image scales to fit, then pads.

        # Base image factore (currently only being resized to final_patch_division_size)
        base_scale_w_factor = final_patch_division_size / original_w
        base_scale_h_factor = final_patch_division_size / original_h

        if scale_w_factor < scale_h_factor:  # Width will be filled completely, height scaled and padded
            scaled_to_w = best_processing_resolution_w
            scaled_to_h = min(math.ceil(original_h * scale_w_factor), best_processing_resolution_h)
        else:  # Height will be filled completely, width scaled and padded
            scaled_to_h = best_processing_resolution_h
            scaled_to_w = min(math.ceil(original_w * scale_h_factor), best_processing_resolution_w)

        # Coordinates on the scaled base image (non anyres), thats currently only being resized
        x_on_base_img = x_pixel_orig * base_scale_w_factor
        y_on_base_img = y_pixel_orig * base_scale_h_factor

        # Find the patch based indices
        base_patch_col = np.clip(int(x_on_base_img // patch_size), 0, final_patch_division_size // patch_size - 1)
        base_patch_row = np.clip(int(y_on_base_img // patch_size), 0, final_patch_division_size // patch_size - 1)
        total_patches_in_row = final_patch_division_size // patch_size
        base_img_idx = base_patch_row * total_patches_in_row + base_patch_col
        final_base_img_idx = image_token_start_index_in_embeds + base_img_idx
        base_xy_coords.append((base_patch_col, base_patch_row))

        resized_mask[base_patch_row, base_patch_col] = 1.0  # Mark the base image patch in the mask

        # Coordinates on the scaled (but not yet padded) image
        x_on_scaled_img = (x_pixel_orig / original_w) * scaled_to_w
        y_on_scaled_img = (y_pixel_orig / original_h) * scaled_to_h

        # Calculate padding added to reach best_processing_resolution
        # padding_left = (best_processing_resolution_w - scaled_to_w) // 2
        # padding_top = (best_processing_resolution_h - scaled_to_h) // 2

        # Final coordinates on the padded image that gets divided into patches
        # x_on_padded_img = x_on_scaled_img + padding_left
        # y_on_padded_img = y_on_scaled_img + padding_top

        # Ensure coordinates are within the bounds of the padded image
        # x_on_padded_img = max(0, min(x_on_padded_img, best_processing_resolution_w - 1))
        # y_on_padded_img = max(0, min(y_on_padded_img, best_processing_resolution_h - 1))

        # 3. Determine which patch these coordinates fall into. Patches are taken row-wise.
        # `divide_to_patches` divides the `image_padded` (which is at best_processing_resolution)
        # into `final_patch_division_size` x `final_patch_division_size` patches.

        x_on_patched_img = x_on_scaled_img // patch_size
        y_on_patched_img = y_on_scaled_img // patch_size
        anyres_patched_row_size = scaled_to_w // patch_size + 1         # +1 for a newline token at the end

        patch_1d_index_on_patched_img = y_on_patched_img * anyres_patched_row_size + x_on_patched_img

        # since the first anyres patch is the base image, we need to add it to the final index
        final_anyres_idx = image_token_start_index_in_embeds + (final_patch_division_size // patch_size)**2 + patch_1d_index_on_patched_img
        final_anyres_idx = int(final_anyres_idx)
        # anyres_patch_col_idx = int(x_on_padded_img // final_patch_division_size)
        # anyres_patch_row_idx = int(y_on_padded_img // final_patch_division_size)
        #
        # num_anyres_patches_per_row_on_padded_img = best_processing_resolution_w // final_patch_division_size
        # num_anyres_patches_per_col_on_padded_img = best_processing_resolution_h // final_patch_division_size
        # # Clamp to valid patch indices on the padded image
        # # Max col index is num_patches_per_row_on_padded_img - 1
        # # Max row index is (best_processing_resolution_h // final_patch_division_size) - 1
        # anyres_patch_col_idx = min(anyres_patch_col_idx, num_anyres_patches_per_row_on_padded_img - 1)
        # anyres_patch_row_idx = min(anyres_patch_row_idx, num_anyres_patches_per_col_on_padded_img - 1)
        #
        # # Convert 2D patch index (on the padded image) to 1D flattened index
        # patch_1d_index_on_padded_img = (anyres_patch_row_idx * num_anyres_patches_per_row_on_padded_img + anyres_patch_col_idx) * final_patch_division_size
        #
        # # 4. Convert to token index in inputs_embeds.
        # # The first token is the global image, then the patches from the padded image.
        # # So, add 1 to the patch_1d_index_on_padded_img.
        # final_token_idx = image_token_start_index_in_embeds + 1 + patch_1d_index_on_padded_img
        # token_indices.append(final_anyres_idx)
        token_indices.append(final_base_img_idx)

    # remove duplicates
    token_indices = np.unique(np.array(token_indices))
    # added constant bias
    token_indices = token_indices + bias_offset
    token_indices = token_indices.tolist()
    # lets also add the start text tokens to the mask (initial prompt text)
    if add_system_prompt_tokens:
        for i in range(0, system_prompt_len):
            token_indices.append(i)
    if add_user_prompt_tokens:
        if isinstance(user_prompt_range, int):
            user_prompt_len = user_prompt_range
            for i in range(system_prompt_len, system_prompt_len + user_prompt_len):
                token_indices.append(i)
        elif isinstance(user_prompt_range, list) and len(user_prompt_range) == 2:
            for i in range(user_prompt_range[0], user_prompt_range[1]):
                token_indices.append(i)

    # for i in range(0, 26):
        # token_indices.append(i)
    # build a mask with the base_xy_coords
    # mask = np.zeros([best_processing_resolution_h, best_processing_resolution_w], dtype=np.float32)
    # base_xy_coords_np = np.c_[base_xy_coords]
    # mask[base_xy_coords_np[:, 1], base_xy_coords_np[:, 0]] = 1.0


    return sorted(list(set(token_indices))), resized_mask


def _pixel_to_token_indices_helper_anyres_inference(
    pixel_coords: List[Tuple[int, int]],
    original_image_size: Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
    final_patch_division_size: int = 384,
    image_token_start_index_in_embeds: int = 14,
    patch_size: int = 14,
    add_system_prompt_tokens: bool = False,
    add_user_prompt_tokens: bool = False,
    user_prompt_len: Optional[int] = 20,
    system_prompt_len: Optional[int] = 14,
    user_prompt_range: Optional[Union[int, Tuple[int, int]]] = [1849, 1869],
    bias_offset: int = 0,
) -> Tuple[List[int], np.ndarray]:
    """
    Alternative implementation that mirrors the actual inference-time preprocessing.

    The pipeline (see ``llava/mm_utils.py``) first selects ``best_processing_resolution``,
    resizes the image while preserving aspect ratio, pads it to the target canvas, splits
    that canvas into ``final_patch_division_size`` (e.g. 384) crops, encodes each crop with a
    ViT of ``patch_size`` (e.g. 14), then flattens the resulting token grid row-wise while
    inserting a newline token after every row. With ``spatial_unpad`` the padded tokens are
    removed before the newline column is added. The very first tokens correspond to the base
    image that is resized directly to ``final_patch_division_size``.
    """
    original_w, original_h = original_image_size
    best_w, best_h = select_best_resolution((original_w, original_h), possible_resolutions)

    # Base image (resized square) scaling factors
    base_scale_w = final_patch_division_size / max(original_w, 1)
    base_scale_h = final_patch_division_size / max(original_h, 1)

    # Determine how the anyres preprocessing resized & padded the image.
    scale_w_factor = best_w / max(original_w, 1)
    scale_h_factor = best_h / max(original_h, 1)
    if scale_w_factor < scale_h_factor:
        scaled_w = best_w
        scaled_h = min(int(math.ceil(original_h * scale_w_factor)), best_h)
    else:
        scaled_h = best_h
        scaled_w = min(int(math.ceil(original_w * scale_h_factor)), best_w)

    # Token-grid bookkeeping (matches spatial_unpad merge)
    tokens_per_patch_side = max(final_patch_division_size // patch_size, 1)
    base_patch_tokens = tokens_per_patch_side ** 2
    anyres_base_offset = image_token_start_index_in_embeds + base_patch_tokens

    # Tokens that survive unpadding (derived from the resized, pre-pad image)
    unpadded_width_tokens = max(int(math.ceil(scaled_w / patch_size)), 1)
    unpadded_height_tokens = max(int(math.ceil(scaled_h / patch_size)), 1)
    anyres_row_stride = unpadded_width_tokens + 1  # +1 for newline token appended per row

    token_indices = set()
    resized_mask = np.zeros(
        [final_patch_division_size // patch_size, final_patch_division_size // patch_size],
        dtype=np.float32,
    )

    for y_pixel_orig, x_pixel_orig in pixel_coords:
        # Base image token (direct resize to final_patch_division_size)
        x_on_base = x_pixel_orig * base_scale_w
        y_on_base = y_pixel_orig * base_scale_h

        base_patch_col = int(np.clip(x_on_base // patch_size, 0, tokens_per_patch_side - 1))
        base_patch_row = int(np.clip(y_on_base // patch_size, 0, tokens_per_patch_side - 1))
        base_linear_idx = base_patch_row * tokens_per_patch_side + base_patch_col
        final_base_idx = image_token_start_index_in_embeds + base_linear_idx
        token_indices.add(int(final_base_idx))
        resized_mask[base_patch_row, base_patch_col] = 1.0

        # Anyres token: map pixel -> scaled (pre-pad), then place on the unpadded grid
        x_scaled = (x_pixel_orig / max(original_w, 1)) * max(scaled_w - 1, 1)
        y_scaled = (y_pixel_orig / max(original_h, 1)) * max(scaled_h - 1, 1)

        token_col = int(np.clip(x_scaled // patch_size, 0, unpadded_width_tokens - 1))
        token_row = int(np.clip(y_scaled // patch_size, 0, unpadded_height_tokens - 1))

        anyres_linear_idx = int(anyres_base_offset + token_row * anyres_row_stride + token_col)
        token_indices.add(anyres_linear_idx)

    # Make deterministic & apply optional offsets/prompts
    token_indices = sorted(token_indices)
    token_indices = (np.array(token_indices) + bias_offset).tolist()

    if add_system_prompt_tokens:
        for i in range(system_prompt_len):
            token_indices.append(i)
    if add_user_prompt_tokens:
        if isinstance(user_prompt_range, int):
            user_prompt_len = user_prompt_range
            for i in range(system_prompt_len, system_prompt_len + user_prompt_len):
                token_indices.append(i)
        elif isinstance(user_prompt_range, (list, tuple)) and len(user_prompt_range) == 2:
            for i in range(user_prompt_range[0], user_prompt_range[1]):
                token_indices.append(i)

    return sorted(set(token_indices)), resized_mask
