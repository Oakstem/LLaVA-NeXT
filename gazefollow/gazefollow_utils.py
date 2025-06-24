import math
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
        base_patch_col = int(x_on_base_img // patch_size)
        base_patch_row = int(y_on_base_img // patch_size)
        total_patches_in_row = final_patch_division_size // patch_size
        base_img_idx = base_patch_row * total_patches_in_row + base_patch_col
        final_base_img_idx = image_token_start_index_in_embeds + base_img_idx
        base_xy_coords.append((base_patch_col, base_patch_row))

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

    # lets also add the start text tokens to the mask (initial prompt text)
    # for i in range(0, 26):
        # token_indices.append(i)

    return sorted(list(set(token_indices))), base_xy_coords
