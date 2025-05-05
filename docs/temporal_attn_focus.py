import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from docs.research_utils import fix_wsl_paths

result_dir = r'D:\Projects\LLaVA-NeXT\llava_attention_maps\00080697_attn_sweep\20250430_170329_Lets_count_one_by_one_the_peop\layer_23'
result_dir = r'D:\Projects\LLaVA-NeXT\llava_attention_maps\00080697_attn_sweep\20250502_145759_You_are_an_expert_vision_assis\layer_23'
result_dir = r'D:\Projects\LLaVA-NeXT\llava_attention_maps\00011001_attn_sweep\20250502_230912_You_are_an_expert_vision_assis\layer_23'
original_img_path = r"D:\Projects\data\gazefollow\train\00000080\00080697.jpg"
original_img_path = r"D:\Projects\data\gazefollow\train\00000011\00011001.jpg"
if 'wsl' in os.uname().release.lower():
    result_dir = fix_wsl_paths(result_dir)
    original_img_path = fix_wsl_paths(original_img_path)
attn_maps = Path(result_dir) / 'attention_tensors'
attn_map_files = sorted(attn_maps.glob("*.pt"))

def find_persons_and_gaze_targets(text_array):
    """
    Find the indices of "Person #:" patterns and their corresponding "looking at" phrases
    in a tokenized array.

    Args:
        text_array (list): List of [index, token_text] pairs

    Returns:
        dict: Dictionary mapping person numbers to a dict containing:
              - 'person_indices': List of indices for the person mention
              - 'gaze_target_indices': List of indices for the gaze target mention (if found)
    """
    persons_data = {}
    i = 0

    # First, find all person references
    while i < len(text_array) - 2:  # Need at least 3 tokens for "Person", number, and colon
        # Check for "Person" token (case insensitive)
        if re.match(r'(?i)person', text_array[i][1]):
            # Check if next tokens contain a number
            potential_number = ""
            number_idx = i + 1

            # Look ahead to find the number (might be in the same or next token)
            if re.search(r'\d+', text_array[i][1]):
                # Number in the same token as "Person"
                match = re.search(r'(?i)person\s*(\d+)', text_array[i][1])
                if match:
                    potential_number = match.group(1)
                    number_idx = i
            else:
                # Check next token(s) for numbers
                for j in range(i + 1, min(i + 4, len(text_array))):
                    if re.search(r'\d+', text_array[j][1]):
                        potential_number = potential_number + re.search(r'(\d+)', text_array[j][1]).group(1)
                        number_idx = j
                        # break

            # If we found a number, look for the colon
            if potential_number:
                person_num = int(potential_number)

                # Look for colon in nearby tokens
                for k in range(number_idx, min(number_idx + 3, len(text_array))):
                    if ':' in text_array[k][1]:
                        # Found the full pattern, store indices
                        indices = [text_array[idx][0] for idx in range(i, k + 1)]
                        persons_data[person_num] = {
                            'person_indices': indices,
                            'gaze_target_indices': []  # Initialize with empty list
                        }
                        i = k  # Skip to after the colon
                        break
                    # check if the next token is a number
                    # elif re.search(r'\d+', text_array[k][1]):
                    #     # append the number to the person number
                    #     potential_number = re.search(r'(\d+)', text_array[k][1]).group(1)

        i += 1

    # Now, look for "looking at" phrases and associate them with the nearest preceding person
    i = 0
    last_person = None

    while i < len(text_array) - 1:
        # Check if the current token is part of a person mention
        for person_num, data in persons_data.items():
            if text_array[i][0] in data['person_indices']:
                last_person = person_num
                break

        current_token = text_array[i][1].lower()

        # Check if current token contains "looking"
        if "looking" in current_token and last_person is not None:
            start_idx = i
            phrase_indices = [text_array[i][0]]

            # Check if "at" is in the same token or the next one
            if "at" in current_token:
                # "looking at" in the same token
                persons_data[last_person]['gaze_target_indices'] = phrase_indices
            elif i + 1 < len(text_array) and "at" in text_array[i + 1][1].lower():
                # "at" is in the next token
                phrase_indices.append(text_array[i + 1][0])
                persons_data[last_person]['gaze_target_indices'] = phrase_indices
                i += 1  # Skip the next token since we've processed it

        i += 1

    return persons_data



#%% Lets Load the attention maps
attn_maps = [torch.load(str(attn_map)) for attn_map in attn_map_files]

#%%
full_text_arr = [[ind, val['token_text']] for ind, val in enumerate(attn_maps)]
full_text_raw = ' '.join([val[1] for val in full_text_arr])
full_text_inds = [val[0] for val in full_text_arr]
# Find the indices of the person in the caption
persons_data = find_persons_and_gaze_targets(full_text_arr)
#%%
# Lets collect all the tokens for each person, and their gaze targets
all_person_indices = [val['person_indices'] for val in persons_data.values()]
for person_num, data in persons_data.items():
    person_indices = data['person_indices']
    gaze_target_indices = data['gaze_target_indices']
    person_start_ind = person_indices[0]
    person_end_ind = gaze_target_indices[0]
    # To find the end of the gaze target description, we will look for '.', '\n' or the start of the next person index
    gaze_target_end = len(full_text_arr) # Initialize to the end of the array
    for gaze_target_ind in range(person_end_ind, len(full_text_arr)):
        if full_text_arr[gaze_target_ind][1] in ['.', '\n'] or gaze_target_ind in all_person_indices:
            gaze_target_end = gaze_target_ind - 1
            break
    # Get the tokens for the person
    person_token_inds = np.arange(person_start_ind, person_end_ind)
    # Get the tokens for the gaze target
    gaze_target_token_inds = np.arange(person_end_ind+1, gaze_target_end + 2)       # skip 'looking' as it mostly gets attention of the person
    # clip the values to the length of the array
    person_token_inds = person_token_inds[person_token_inds < len(full_text_arr)]
    gaze_target_token_inds = gaze_target_token_inds[gaze_target_token_inds < len(full_text_arr)]
    # Join the tokens to form a string
    person_tokens = ' '.join([full_text_arr[ind][1] for ind in person_token_inds if len(full_text_arr[ind]) >= 2])
    gaze_target_tokens = ' '.join([full_text_arr[ind][1] for ind in gaze_target_token_inds if len(full_text_arr[ind]) >= 2])
    # Store the tokens in the dictionary
    persons_data[person_num]['person_tokens'] = person_tokens
    persons_data[person_num]['gaze_target_tokens'] = gaze_target_tokens
    persons_data[person_num]['person_indices'] = person_token_inds
    persons_data[person_num]['gaze_target_indices'] = gaze_target_token_inds


#%% For each person, lets get the attention maps
def get_person_attention_maps(persons_data, attn_maps):
    person_attn = {}
    gaze_target_attn = {}
    for person_num, data in persons_data.items():
        person_attn[person_num] = []
        gaze_target_attn[person_num] = []
        # Get attention maps for person tokens
        for ind in data['person_indices']:
            if isinstance(ind, (int, np.integer)) and ind < len(attn_maps):
                person_attn[person_num].append(attn_maps[ind])

        # Get attention maps for gaze target tokens
        for ind in data['gaze_target_indices']:
            if isinstance(ind, (int, np.integer)) and ind < len(attn_maps):
                gaze_target_attn[person_num].append(attn_maps[ind])

    return person_attn, gaze_target_attn

person_attn, gaze_target_attn = get_person_attention_maps(persons_data, attn_maps)

# average the attention maps
person_attn_avg = {}
gaze_target_attn_avg = {}
for person_attn_i, gaze_attn_i in zip(person_attn.items(), gaze_target_attn.items()):
    person_ind = person_attn_i[0]
    all_attn = [data['attention_map'] for data in person_attn_i[1]]
    person_attn_avg[person_ind] = np.mean(np.stack(all_attn, axis=0), axis=0)

    all_gaze_attn = [data['attention_map'] for data in gaze_attn_i[1]]
    gaze_target_attn_avg[person_ind] = np.mean(np.stack(all_gaze_attn, axis=0), axis=0)

#%% Lets visualize the attention maps
def visualize_attention_map(attn_map, img_path, save_path, th=0.002):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # threshold the attention map
    attn_map_filt = attn_map.copy()
    attn_map_filt[attn_map < th] = 0
    # normalize attn map
    # attn_map_filt = (attn_map_filt - np.min(attn_map_filt)) / (np.max(attn_map_filt) - np.min(attn_map_filt))
    const_factor = 500
    attn_map_filt = attn_map_filt * const_factor
    attn_map_filt = (attn_map_filt * 255).astype(np.uint8)
    # Resize the attention map to match the image size
    attn_map_resized = cv2.resize(attn_map_filt, (img.shape[1], img.shape[0]))

    # Create a heatmap from the attention map
    heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlayed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Save the overlayed image
    cv2.imwrite(save_path, overlayed_img)
    print(f"Saved attention map to {save_path}")
#%% Visualize each person attn maps
# # lets save the attention maps for each person
# for person_id, val in person_attn.items():
#     for person_data in val:
#         save_path = Path(result_dir) / 'each_person_attn_maps' / f"person_{person_id}" / f"{person_data['token_text']}.png"
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         visualize_attention_map(person_data['attention_map'], original_img_path, save_path)

#%% Lets visualize the attention maps
# def overlay_attn(img_path, attn_map, th=0.002):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # threshold the attention map
#     attn_map_filt = attn_map.copy()
#     attn_map_filt[attn_map < th] = 0
#     # normalize attn map
#     # attn_map_filt = (attn_map_filt - np.min(attn_map_filt)) / (np.max(attn_map_filt) - np.min(attn_map_filt))
#     # attn_map_filt = (attn_map_filt * 255).astype(np.uint8)
#     const_factor = 100000
#     attn_map_filt = attn_map_filt * const_factor
#     attn_map_filt_u8 = (attn_map_filt).astype(np.uint8)
#     # Resize the attention map to match the image size
#     attn_map_resized = cv2.resize(attn_map_filt_u8, (img.shape[1], img.shape[0]))
#
#     # Create a heatmap from the attention map
#     heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
#
#     # Overlay the heatmap on the original image
#     overlayed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
#     return overlayed_img
#
# th = 0.0012
# gaze_th = 0.0001
# for i, attn_map in person_attn_avg.items():
#     gaze_attn_map = gaze_target_attn_avg.get(i, None)
#     save_path = Path(result_dir) / f"person_{i}_attn_map.png"
#     gaze_save_path = Path(result_dir) / f"gaze_target_{i}_attn_map.png"
#     # visualize_attention_map(attn_map, original_img_path, str(save_path))
#     img_path = original_img_path
#     save_path = str(save_path)
#
#     # Overlay the attention map on the image
#     overlayed_img = overlay_attn(img_path, attn_map, th=th)
#     if gaze_attn_map is not None:
#         gaze_overlayed_img = overlay_attn(img_path, gaze_attn_map, th=gaze_th)
#         cv2.imwrite(gaze_save_path, gaze_overlayed_img)
#         print(f"Saved gaze attention map to {gaze_save_path}")
#
#     # Save the overlayed image
#     cv2.imwrite(save_path, overlayed_img)
#     print(f"Saved attention map to {save_path}")
#


#%%
def smooth_and_aggregate_attention_points(attn_map, threshold=0.0012, kernel_size=3, sigma=2):
    """
    Smooth the attention map using a Gaussian filter and extract aggregated points.

    Args:
        attn_map: The attention map as a numpy array
        threshold: Minimum threshold for considering points
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        centers: Aggregated center points
        smoothed_map: The smoothed attention map
    """
    import scipy.ndimage as ndimage
    from sklearn.cluster import MeanShift, estimate_bandwidth

    # 1. Smooth the attention map with a Gaussian filter
    smoothed_map = ndimage.gaussian_filter(attn_map.astype(np.float32), sigma=sigma)

    # 2. Find points above threshold in smoothed map
    points = np.argwhere(smoothed_map > threshold)

    if len(points) == 0:
        return [], smoothed_map

    # Convert to (x, y) format for clustering
    points = points[:, ::-1]

    # 3. Use Mean Shift clustering to find centroids
    # Estimate bandwidth if we have enough points
    if len(points) > 10:
        print(f"Estimating bandwidth for {len(points)} points")
        bandwidth = estimate_bandwidth(points, quantile=0.1)
    else:
        print(f"Not enough points for bandwidth estimation, using kernel size {kernel_size}")
        bandwidth = kernel_size

    # Apply mean shift clustering
    if bandwidth > 0:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(points)
        centers = ms.cluster_centers_.astype(int)
    else:
        # If bandwidth estimation fails, fall back to the points
        centers = points

    return centers, smoothed_map


# Example usage:
def visualize_attention_with_centers(img_path, attn_map, centers, save_path, threshold=0.0012):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare attention map for visualization
    attn_map_filt = attn_map.copy()
    attn_map_filt[attn_map < threshold] = 0
    const_factor = 100000
    attn_map_filt = attn_map_filt * const_factor
    attn_map_filt_u8 = attn_map_filt.astype(np.uint8)

    # Resize to match image dimensions
    attn_map_resized = cv2.resize(attn_map_filt_u8, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)

    # Combine image and heatmap
    overlayed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Resize centers to match image dimensions
    resize_factor = np.array(img.shape[:2]) / np.array(attn_map.shape[:2])
    resize_factor = resize_factor[::-1]  # Reverse for width and height

    if len(centers) == 0:
        print("No centers found above the threshold.")
        resized_centers = centers
    else:
        resized_centers = (centers * resize_factor).astype(int)
    # Draw centers on the image
    for center in centers:
        # Convert center to image coordinates
        center_img = (center * resize_factor).astype(int)
        cv2.circle(overlayed_img, tuple(center_img), 10, (0, 255, 0), -1)

    # Save the result
    cv2.imwrite(save_path, overlayed_img)
    print(f"Saved attention map with centers to {save_path}")

    return resized_centers, overlayed_img


th = 0.0007
person_smooth_attn = {}
# Now let's apply this to each person's attention map
# Process both person and gaze attention maps
for i in person_attn_avg.keys():
    # Process person attention maps
    print("####" * 100)
    print(f"Processing person {i} with attention map shape: {person_attn_avg[i].shape}")
    # Get aggregated centers and smoothed map for person
    p_centers, p_smoothed_map = smooth_and_aggregate_attention_points(person_attn_avg[i], threshold=th, sigma=1)
    person_smooth_attn[i] = {'centers': p_centers, 'smoothed_map': p_smoothed_map}
    # Save visualization for person
    p_save_path = Path(result_dir) / f"person_{i}_attn_map_smooth_centers.png"
    p_resized_centers, p_overlayed_img = visualize_attention_with_centers(original_img_path, p_smoothed_map, p_centers, str(p_save_path), threshold=th)
    # Save the centers and the smoothed map to a file for person
    person_results = {
        'centers': p_resized_centers,
        'smoothed_map': p_smoothed_map
    }
    p_save_path = Path(result_dir) / 'each_person_attn_maps' / f"person_{i}" / f"person_{i}_attn_map_smooth_centers.pt"
    p_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(person_results, str(p_save_path))
    print(f"Person {i} has {len(p_centers)} attention centers saved to {p_save_path}")

    # Process gaze attention maps
    if i in gaze_target_attn_avg:
        print(f"Processing gaze for person {i} with attention map shape: {gaze_target_attn_avg[i].shape}")
        # Get aggregated centers and smoothed map for gaze
        g_centers, g_smoothed_map = smooth_and_aggregate_attention_points(gaze_target_attn_avg[i], threshold=0.0001, kernel_size=2, sigma=2)
        # Save visualization for gaze
        g_save_path = Path(result_dir) / f"gaze_target_{i}_attn_map_smooth_centers.png"
        g_resized_centers, g_overlayed_img = visualize_attention_with_centers(original_img_path, g_smoothed_map, g_centers, str(g_save_path), threshold=0.)
        # Save the centers and the smoothed map to a file for gaze
        gaze_results = {
            'centers': g_resized_centers,
            'smoothed_map': g_smoothed_map
        }
        g_save_path = Path(result_dir) / 'each_person_attn_maps' / f"person_{i}" / f"gaze_target_{i}_attn_map_smooth_centers.pt"
        g_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gaze_results, str(g_save_path))
        print(f"Gaze target for person {i} has {len(g_centers)} attention centers saved to {g_save_path}")
    # break
