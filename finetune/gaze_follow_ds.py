import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from docs.research_utils import fix_wsl_paths


# annot_path = r"D:\Projects\data\gazefollow\test_annotations_release.txt"
annot_path = r"D:\Projects\data\gazefollow\train_annotations_release.txt"
base_data_dir_path = r"D:\Projects\data\gazefollow"
llava_results_dir = r"D:\Projects\LLaVA-NeXT\llava_attention_sweep\20250503_001255_You_are_an_expert_vision_assis"
llava_results_dir = Path(fix_wsl_paths(llava_results_dir))
base_data_dir_path = Path(fix_wsl_paths(base_data_dir_path))
annot_path = fix_wsl_paths(annot_path)

#%% useful functions
def filter_gaze_points(points, threshold_factor=1.5):
    """
    Filter outliers from a 2D array of coordinate points using median-based approach.

    Args:
        points: torch.Tensor or numpy array of shape (n, 2) containing x,y coordinates
        threshold_factor: Factor to multiply MAD for determining outlier threshold

    Returns:
        tuple: (filtered_mean, filtered_points)
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = np.array(points)

    # Calculate the median point
    median_point = np.median(points_np, axis=0)

    # Calculate distance of each point from the median point
    distances = np.sqrt(np.sum((points_np - median_point) ** 2, axis=1))

    # Calculate the median absolute deviation (MAD)
    mad = np.median(np.abs(distances - np.median(distances)))

    # Define the threshold (adjusted with the scaling factor)
    threshold = np.median(distances) + threshold_factor * mad

    # Find indices of inliers
    inliers_mask = distances <= threshold
    filtered_points = points_np[inliers_mask]

    # Calculate mean of filtered points
    if len(filtered_points) > 0:
        filtered_mean = np.mean(filtered_points, axis=0)
    else:
        filtered_mean = median_point  # Fallback to median if no inliers

    return filtered_mean, filtered_points

#%% read the annotations
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

#%% to numeric
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
#%% Since every image has several annotations, we need to group the annotations by image and average the gaze points
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

#%% For every row in the compact_df, we need to find the corresponding image in the llava_results_dir
# and load the gaze points

# for ind, row in compact_df.iterrows():
# row = compact_df.iloc[0]
# image_path = row['image_path']
# image_path = image_path.replace("\\", os.sep)
# image_path = image_path.replace("D:", str(base_data_dir_path))
# # get the image name
# image_name = os.path.basename(image_path)
# # get the image directory
# image_dir = os.path.dirname(image_path)
# # get the llava results path
# llava_results_path = llava_results_dir / image_dir / f"{image_name}.json"
# if not llava_results_path.exists():
#     print(f"llava results path does not exist: {llava_results_path}")
#     # continue
# # load the json file
# with open(llava_results_path, 'r') as f:
#     llava_results = json.load(f)

# get the gaze points
gaze_points = llava_results['gaze_points']
#%%
llava_persons_segment = list(llava_results_dir.rglob("*all_segmentation_results.json"))
llava_gaze_points = list(llava_results_dir.rglob("*gaze*attn_map_smooth_centers.pt"))

#%% lets load all the gaze points and aggregate them by image id
gaze_points_dd = {}
for gaze_points_path in tqdm(llava_gaze_points):
    # gaze_points_path = llava_gaze_points[0]
    gaze_filename = gaze_points_path.stem
    # load the gaze points
    gaze_points = torch.load(gaze_points_path)
    # get the gaze points
    gaze_point = torch.load(gaze_points_path)['centers']
    # get the image name
    image_name = [val.split('_')[0] for val in gaze_points_path.parts if val.endswith('_attn')][0]
    person_id = gaze_filename.split('_')[2]
    if image_name not in gaze_points_dd:
        gaze_points_dd[image_name] = {}
    gaze_points_dd[image_name][person_id] = gaze_point

#%% lets apply our median + mean filtering to extract a single center point for each person
gaze_points_dd_extra = gaze_points_dd.copy()
for image_name, gaze_points in tqdm(gaze_points_dd_extra.items()):
    for person_id, gaze_point in gaze_points.items():
        if len(gaze_point) == 0:
            continue
        # filter the gaze points
        mean_point, filtered_points = filter_gaze_points(gaze_point)
        # save the filtered points
        gaze_points_dd_extra[image_name][person_id] = {
            'mean_point': mean_point,
            'filtered_points': filtered_points,
            'original_points': gaze_point
        }
#%% lets go over the persons and load their bboxes
person_bboxes = {}
for ind, row in enumerate(tqdm(llava_persons_segment)):
    image_id = row.parts[7].split('_')[0]
    person_data = json.load(open(row))
    person_bboxes[image_id] = {}
    # rename person_1 to 1
    person_keys = list(person_data.keys())
    for key in person_keys:
        new_key = key.replace('person_', '')
        person_data[new_key] = person_data.pop(key)

    # lets load a segmentation mask to extract the original image dimensions (since gt is in relational coordinates)
    imgs = list(Path(row.parent).rglob("*.jpg"))
    if len(imgs) == 0:
        print(f"no segmentation images found for {row}")
        continue

    # load the image
    img_path = imgs[0]
    img = cv2.imread(str(img_path))
    # get the image dimensions
    h, w, _ = img.shape
    person_data['image_shape'] = (h, w)
    person_bboxes[image_id] = person_data

#%% Now lets go over the GT dataframe and add the resulted gaze points for each frame
def bbox_intersect(bbox1, bbox2):
    # Calculate the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    return intersection_area

def find_matching_person(llava_person_result, gt_person_bbox):
    all_intersects = []
    matched_person_id = None
    for person_id, person_data in llava_person_result.items():
        if 'image_shape' in person_id or 'person' not in person_data:
            continue
        # get the bbox
        person_bbox = person_data['person']['boxes'][0]
        # calculate the iou
        intersect = bbox_intersect(gt_person_bbox, person_bbox)
        # if the iou is greater than 0.5, we have a match
        if intersect > 0:
            all_intersects.append([intersect, person_id])
    if len(all_intersects) == 0:
        print(f"no intersecting persons found for {image_id}")
        return matched_person_id
    all_intersects = np.array(all_intersects)
    try:
        matched_person_id = all_intersects[np.argmax(all_intersects[:,0])][1]
    except ValueError:
        return all_intersects
    return matched_person_id

# lets add additional columns to the compact_df
compact_df['llava_matched_person_id'] = None
compact_df['llava_person_bbox'] = None
compact_df['llava_gaze_points'] = None
compact_df['gaze_error'] = None

for ind, row in tqdm(compact_df.iterrows()):
    image_path = row['image_path']
    image_id = Path(image_path).stem
    llava_gaze_result = gaze_points_dd_extra.get(image_id)
    llava_person_result = person_bboxes.get(image_id)
    if llava_gaze_result is None or llava_person_result is None:
        print(f"no llava gaze result found for {image_id}")
        continue
    h, w = llava_person_result['image_shape']
    gt_person_bbox = [w*row['body_bbox_x'], h*row['body_bbox_y'],
                      w*(row['body_bbox_x']+row['body_bbox_width']), h*(row['body_bbox_y']+row['body_bbox_height'])]
    llava_matched_person_id = find_matching_person(llava_person_result, gt_person_bbox)
    if llava_matched_person_id is None:
        print(f"no matched person found for {image_id}")
        continue
    row['llava_matched_person_id'] = llava_matched_person_id
    # row['llava_person_bbox'] = llava_person_result.get(llava_matched_person_id, {}).get('person')['boxes'][0]
    # row['llava_gaze_points'] =
    llava_person_bbox = llava_person_result.get(llava_matched_person_id, {}).get('person')['boxes'][0]
    llava_gaze_point = llava_gaze_result.get(llava_matched_person_id, {})
    if llava_person_bbox is None or len(llava_gaze_point) == 0:
        print(f"no gaze points found for {image_id}")
        continue
    row['llava_gaze_points'] = llava_gaze_point['mean_point']
    row['llava_person_bbox'] = llava_person_bbox
    # calculate the distance between the gt gaze point and the llava gaze point
    gt_gaze_point = np.array([row['gaze_x'], row['gaze_y']])
    llava_gaze_point = np.array(row['llava_gaze_points']) / np.array([w, h])        # predicted coordinates are in [x, y]
    # calculate the distance
    distance = np.linalg.norm(gt_gaze_point - llava_gaze_point)
    row['gaze_error'] = distance
    # set the df with the new row
    compact_df.iloc[ind] = row

# print the average gaze error
print(f"average gaze error: {compact_df['gaze_error'].mean()}")
#%% Save the results to a csv file
output_path = llava_results_dir / "gaze_follow_results.csv"
compact_df.to_csv(output_path, index=False)
print(f"saved the results to {output_path}")
#%%
points_path = fix_wsl_paths(r"D:\Projects\LLaVA-NeXT\llava_attention_sweep\20250503_001255_You_are_an_expert_vision_assis\00000001_attn\layer_23\each_person_attn_maps\person_1\gaze_target_1_attn_map_smooth_centers.pt")
gaze_points = torch.load(points_path)
