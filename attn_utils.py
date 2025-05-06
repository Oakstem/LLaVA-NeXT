import os
import re
import cv2
import torch
import numpy as np
from pathlib import Path
import scipy.ndimage as ndimage
from sklearn.cluster import MeanShift, estimate_bandwidth

def fix_wsl_paths(path):
    """Fix Windows paths for WSL environment"""
    if not isinstance(path, str):
        return path

    path = path.replace('\\', '/')
    if path.startswith('D:'):
        path = '/mnt/d' + path[2:]
    return path

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

            # If we found a number, look for the colon
            if potential_number:
                person_num = int(potential_number)

                # Look for colon in nearby tokens
                for k in range(number_idx, min(number_idx + 3, len(text_array))):
                    if ':' in text_array[k][1]:
                        # Found the full pattern, store indices
                        indices = [text_array[idx][0] for idx in range(k, k + 1)]
                        persons_data[person_num] = {
                            'person_indices': indices,
                            'gaze_target_indices': []  # Initialize with empty list
                        }
                        i = k  # Skip to after the colon
                        break
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
            phrase_indices = [text_array[i+2][0]]       # skip "looking" and "at"

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

def process_persons_data(persons_data, full_text_arr):
    """Process person data to extract token information"""
    all_person_indices = [val['person_indices'][0] for val in persons_data.values()]

    for person_num, data in persons_data.items():
        person_indices = data['person_indices']
        gaze_target_indices = data['gaze_target_indices']

        if not gaze_target_indices:
            continue

        person_start_ind = person_indices[0] + 1  # start after ':'
        person_end_ind = gaze_target_indices[0] - 1     # gaze target starts after "looking at"

        # To find the end of the gaze target description, look for '.', '\n' or the start of the next person index
        gaze_target_end = len(full_text_arr)  # Initialize to the end of the array
        for gaze_target_ind in range(person_end_ind, len(full_text_arr)):
            if full_text_arr[gaze_target_ind][1] in ['.', '\n'] or gaze_target_ind in all_person_indices:
                if gaze_target_ind in all_person_indices:
                    # If we hit another person (":"), rewind 3 tokens
                    gaze_target_end = gaze_target_ind - 3
                else:
                    gaze_target_end = gaze_target_ind
                break

        # Get the tokens for the person
        person_token_inds = np.arange(person_start_ind, person_end_ind)     # start person at ":" and end at "looking"
        # Get the tokens for the gaze target
        gaze_target_token_inds = np.arange(person_end_ind+1, gaze_target_end)     # start gaze target at "at"+1 and end at "." | "\n" | next person

        # Clip the values to the length of the array
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

    return persons_data

def get_person_attention_maps(persons_data, attn_maps):
    """Extract attention maps for each person and gaze target"""
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

def average_attention_maps(person_attn, gaze_target_attn):
    """Average the attention maps for each person and gaze target"""
    person_attn_avg = {}
    gaze_target_attn_avg = {}

    for person_ind, attn_maps in person_attn.items():
        if attn_maps:
            all_attn = [data['attention_map'] for data in attn_maps]
            person_attn_avg[person_ind] = np.mean(np.stack(all_attn, axis=0), axis=0)

    for person_ind, attn_maps in gaze_target_attn.items():
        if attn_maps:
            all_gaze_attn = [data['attention_map'] for data in attn_maps]
            gaze_target_attn_avg[person_ind] = np.mean(np.stack(all_gaze_attn, axis=0), axis=0)

    return person_attn_avg, gaze_target_attn_avg

def smooth_and_aggregate_attention_points(attn_map, threshold=0.0012, kernel_size=3, sigma=2, adaptive_th=True, adaptive_quant=0.97):
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
    # 1. Smooth the attention map with a Gaussian filter
    smoothed_map = ndimage.gaussian_filter(attn_map.astype(np.float32), sigma=sigma)

    # 2. Find points above threshold in smoothed map
    if adaptive_th:
        # Use quantile to set threshold adaptively
        threshold = np.quantile(smoothed_map, adaptive_quant)
        # clip the minimum
        threshold = np.clip(threshold, 0.00007, 1)
    points = np.argwhere(smoothed_map > threshold)

    if len(points) == 0:
        return [], smoothed_map

    # Convert to (x, y) format for clustering
    points = points[:, ::-1]

    # 3. Use Mean Shift clustering to find centroids
    # Estimate bandwidth if we have enough points
    if len(points) > 10:
        bandwidth = estimate_bandwidth(points, quantile=0.1)
    else:
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

def visualize_attention_with_centers(img_path, attn_map, centers, save_path, threshold=0.0012):
    """Visualize attention map with identified centers"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

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
        resized_centers = []
    else:
        resized_centers = (centers * resize_factor).astype(int)

    # Draw centers on the image
    for center in resized_centers:
        cv2.circle(overlayed_img, tuple(center), 10, (0, 255, 0), -1)

    # Save the result
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))

    return resized_centers, overlayed_img

def visualize_attention_map(attn_map, img_path, save_path, threshold=0.002):
    """Basic visualization of attention map without centers"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Threshold the attention map
    attn_map_filt = attn_map.copy()
    attn_map_filt[attn_map < threshold] = 0

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
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
