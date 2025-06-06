import cv2
import pandas as pd
from pathlib import Path
import numpy as np # Retained for pd.isna checks if target_row values could be np.nan
from gazefollow.gazefollow_utils import prepare_gaze_follow_dataset
from docs.research_utils import fix_wsl_paths

def visualize_gaze_entry(image_id_to_show: str, df_to_use: pd.DataFrame, data_base_dir: Path, additional_bboxes: list = None):
    """
    Visualizes the ground truth gaze data for a given image ID from the GazeFollow dataset.

    Args:
        image_id_to_show: The stem of the image file (e.g., "00000001").
        df_to_use: The DataFrame containing the gaze data (e.g., compact_df).
        data_base_dir: The base directory where images are stored.
        additional_bboxes: List of additional bounding boxes to draw on the image.
    """
    target_row = None
    # Ensure image_path is available for iteration
    if 'image_path' not in df_to_use.columns:
        print("Error: 'image_path' column not found in DataFrame.")
        return

    target_row = df_to_use[df_to_use['image_path'] == image_id_to_show]
    if target_row.shape[0] >= 1:
        print(f"Warning: Expected 1 row for image ID {image_id_to_show}, but got {target_row.shape[0]} rows.")
        target_row = target_row.iloc[0, :]

    if target_row is None or target_row.shape[0] == 0:
        print(f"Image ID {image_id_to_show} not found in the DataFrame.")
        return

    # Construct image path and load image


    full_image_path = data_base_dir / target_row['image_path']

    if not full_image_path.exists():
        print(f"Attempting to locate image: {full_image_path}")
        # Fallback 1: Try 'original_path' if it exists and is valid
        if 'original_path' in target_row and not pd.isna(target_row['original_path']):
            original_path_candidate = Path(target_row['original_path'])
            if original_path_candidate.exists():
                full_image_path = original_path_candidate
                print(f"Found image using 'original_path': {full_image_path}")
            else:
                # Fallback 2: Check if 'image_path' column itself contains an absolute path
                image_path_abs_candidate = Path(target_row['image_path'])
                if image_path_abs_candidate.is_absolute() and image_path_abs_candidate.exists():
                    full_image_path = image_path_abs_candidate
                    print(f"Found image using absolute path from 'image_path' column: {full_image_path}")
                else:
                    print(f"Image file not found at {data_base_dir / target_row['image_path']}, nor via fallbacks.")
                    return
        # Fallback 2 (if 'original_path' not present or was NaN): Check if 'image_path' column itself contains an absolute path
        elif Path(target_row['image_path']).is_absolute() and Path(target_row['image_path']).exists():
            full_image_path = Path(target_row['image_path'])
            print(f"Found image using absolute path from 'image_path' column: {full_image_path}")
        else:
            print(f"Image file not found at {data_base_dir / target_row['image_path']}, nor via fallbacks.")
            return


    img = cv2.imread(str(full_image_path))
    if img is None:
        print(f"Failed to load image: {full_image_path}")
        return

    h, w, _ = img.shape

    # --- Ground Truth Data ---
    # Body Bbox (relative to image size)
    if not all(col in target_row and not pd.isna(target_row[col]) for col in ['body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height']):
        print(f"Missing or NaN body bounding box data for image ID {image_id_to_show}.")
        # Optionally, still display image without bbox or return
    else:
        gt_body_x1 = int(target_row['body_bbox_x'] * w)
        gt_body_y1 = int(target_row['body_bbox_y'] * h)
        gt_body_x2 = int((target_row['body_bbox_x'] + target_row['body_bbox_width']) * w)
        gt_body_y2 = int((target_row['body_bbox_y'] + target_row['body_bbox_height']) * h)
        cv2.rectangle(img, (gt_body_x1, gt_body_y1), (gt_body_x2, gt_body_y2), (0, 255, 0), 2) # Green

    # Gaze Point (relative to image size)
    if not all(col in target_row and not pd.isna(target_row[col]) for col in ['gaze_x', 'gaze_y']):
        print(f"Missing or NaN gaze point data for image ID {image_id_to_show}.")
        # Optionally, still display image without gaze point or return
    else:
        gt_gaze_x = int(target_row['gaze_x'] * w)
        gt_gaze_y = int(target_row['gaze_y'] * h)
        cv2.circle(img, (gt_gaze_x, gt_gaze_y), 10, (0, 255, 0), -1) # Green dot

    # --- Additional Bounding Boxes (Normalized) ---
    if additional_bboxes:
        for bbox in additional_bboxes:
            if len(bbox) == 4: # Assuming (x_min, y_min, x_max, y_max)
                x1_norm, y1_norm, x2_norm, y2_norm = bbox
                add_x1 = int(x1_norm * w)
                add_y1 = int(y1_norm * h)
                add_x2 = int(x2_norm * w)
                add_y2 = int(y2_norm * h)
                cv2.rectangle(img, (add_x1, add_y1), (add_x2, add_y2), (255, 0, 0), 2) # Blue
            else:
                print(f"Skipping invalid additional bounding box: {bbox}. Expected 4 values.")

    # write the image to a file
    output_dir = Path(data_base_dir) / 'gaze_visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{Path(image_id_to_show).stem}.jpg"), img)
    print(f"Saved image to {output_dir / f'{Path(image_id_to_show).stem}.jpg'}")

# Example usage (assuming you have 'compact_df' and 'base_data_dir_path' loaded elsewhere):
if __name__ == '__main__':
    # This is a placeholder for how you might load your data if running this file directly
    # For actual use, import visualize_gaze_entry into your main script (e.g., gaze_follow_ds.py)
    # and call it with your loaded DataFrame and base_data_dir_path.


    # You would need actual images at the specified paths for this example to fully work.
    # Create dummy image files for the example to run without erroring on imread
    # (Path(mock_base_data_dir) / Path(data['image_path'][0])).parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(Path(mock_base_data_dir) / Path(data['image_path'][0])), np.zeros((100,100,3), dtype=np.uint8))
    annot_path = r"D:\Projects\data\gazefollow\train_annotations_release.txt"
    annot_path = r"D:\Projects\data\gazefollow\test_annotations_release.txt"
    data_base_dir = r"D:\Projects\data\gazefollow"
    annot_path = fix_wsl_paths(annot_path)
    data_base_dir = fix_wsl_paths(data_base_dir)
    results_dict = prepare_gaze_follow_dataset(annot_path, data_base_dir)
    df = results_dict['df']
    split_type = 'train' if 'train' in annot_path else 'test2'
    # lets build the image key to be in the format train/00000080/00080697.jpg
    image_key = '00000002'
    image_key = Path(image_key).stem
    image_full_key = f"{split_type}/000{image_key[:-3]}/{image_key}.jpg"

    # Example of additional bounding boxes
    example_additional_bboxes = [(0.152, 0.176, 0.636, 0.934), (0.536, 0.204, 0.784, 0.808),
                                 (0.736, 0.204, 0.952, 0.808)]

    visualize_gaze_entry(image_full_key, df, Path(data_base_dir), additional_bboxes=example_additional_bboxes)
