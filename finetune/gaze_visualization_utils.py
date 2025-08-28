import cv2
import pandas as pd
from pathlib import Path
import numpy as np # Retained for pd.isna checks if target_row values could be np.nan
import ast
import json
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


def visualize_gaze_results_comparison(csv_path: str, output_dir: str = None, max_images: int = None):
    """
    Visualize GazeFollowing results comparing ground truth and predictions with overlays.
    
    Args:
        csv_path: Path to the results CSV file
        output_dir: Directory to save visualization images (default: creates gaze_results_visualization)
        max_images: Maximum number of images to process (default: all)
    """
    # UI Parameters - easy to modify for styling
    TEXT_BOX_WIDTH = 250
    TEXT_BOX_HEIGHT = 85
    TEXT_BOX_X = 10
    TEXT_BOX_Y = 5
    TEXT_START_X = 15
    TEXT_START_Y = 25
    TEXT_LINE_SPACING = 15
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    LEGEND_FONT_SCALE = 0.4
    
    # Fix WSL path
    csv_path = fix_wsl_paths(csv_path)
    
    # Load the results CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent / 'gaze_results_visualization'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    processed_count = 0
    for idx, row in df.iterrows():
        if max_images and processed_count >= max_images:
            break
            
        # Skip if no image path
        if pd.isna(row['image_path']):
            continue
            
        # Construct image path (assuming base path structure)
        image_path = Path(row['image_path'])
        
        # Try different base paths to find the image
        possible_bases = [
            "/mnt/d/Projects/data/gazefollow",
            "/mnt/d/Projects/LLaVA-NeXT/gazefollow",
            Path(csv_path).parent.parent.parent.parent / "data" / "gazefollow"
        ]
        
        full_image_path = None
        for base in possible_bases:
            candidate_path = Path(base) / image_path
            if candidate_path.exists():
                full_image_path = candidate_path
                break
        
        if full_image_path is None:
            print(f"Image not found: {image_path}")
            continue
            
        # Load image
        img = cv2.imread(str(full_image_path))
        if img is None:
            print(f"Failed to load image: {full_image_path}")
            continue
            
        h, w, _ = img.shape
        
        # Create visualization overlay
        overlay = img.copy()
        
        # --- Ground Truth Visualization (Green) ---
        # GT Body bbox
        if not pd.isna(row['body_bbox_x']) and not pd.isna(row['body_bbox_y']):
            gt_x1 = int(row['body_bbox_x'] * w)
            gt_y1 = int(row['body_bbox_y'] * h)
            gt_x2 = int((row['body_bbox_x'] + row['body_bbox_width']) * w)
            gt_y2 = int((row['body_bbox_y'] + row['body_bbox_height']) * h)
            cv2.rectangle(overlay, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 3)  # Green
            cv2.putText(overlay, "GT", (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # GT Eye position (small circle)
        if not pd.isna(row['eye_x']) and not pd.isna(row['eye_y']):
            gt_eye_x = int(row['eye_x'] * w)
            gt_eye_y = int(row['eye_y'] * h)
            cv2.circle(overlay, (gt_eye_x, gt_eye_y), 5, (0, 255, 0), -1)  # Green dot
        
        # GT Gaze target (larger circle)
        if not pd.isna(row['gaze_x']) and not pd.isna(row['gaze_y']):
            gt_gaze_x = int(row['gaze_x'] * w)
            gt_gaze_y = int(row['gaze_y'] * h)
            cv2.circle(overlay, (gt_gaze_x, gt_gaze_y), 15, (0, 255, 0), 3)  # Green circle
            cv2.putText(overlay, "GT Gaze", (gt_gaze_x+20, gt_gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # --- Prediction Visualization (Red) ---
        # Predicted person bbox
        if not pd.isna(row['llava_person_bbox']):
            try:
                # Parse the bbox string (could be list format)
                bbox_str = str(row['llava_person_bbox']).strip()
                if bbox_str.startswith('[') and bbox_str.endswith(']'):
                    pred_bbox = ast.literal_eval(bbox_str)
                    if len(pred_bbox) == 4:
                        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
                        cv2.rectangle(overlay, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 3)  # Red
                        cv2.putText(overlay, "PRED", (pred_x1, pred_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except:
                pass
        
        # Predicted gaze points
        if not pd.isna(row['llava_gaze_points']):
            try:
                # Parse the gaze points string (numpy array format)
                gaze_str = str(row['llava_gaze_points']).strip()
                if gaze_str.startswith('[') and gaze_str.endswith(']'):
                    # Remove brackets and clean up the string
                    content = gaze_str[1:-1].strip()
                    # Split by whitespace and filter out empty strings
                    parts = [x for x in content.split() if x]
                    if len(parts) >= 2:
                        pred_gaze_x = int(float(parts[0]))
                        pred_gaze_y = int(float(parts[1]))
                        cv2.circle(overlay, (pred_gaze_x, pred_gaze_y), 15, (0, 0, 255), 3)  # Red circle
                        cv2.putText(overlay, "PRED Gaze", (pred_gaze_x+20, pred_gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif ' ' in gaze_str:
                    # Handle space-separated format
                    parts = gaze_str.split()
                    if len(parts) >= 2:
                        pred_gaze_x = int(float(parts[0]))
                        pred_gaze_y = int(float(parts[1]))
                        cv2.circle(overlay, (pred_gaze_x, pred_gaze_y), 15, (0, 0, 255), 3)  # Red circle
                        cv2.putText(overlay, "PRED Gaze", (pred_gaze_x+20, pred_gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error parsing gaze points for {row['image_key']}: {e}")
        
        # --- Error Information (Top-left corner) ---
        y_offset = TEXT_START_Y
        
        # Background rectangle for text
        text_bg_color = (0, 0, 0)  # Black background
        cv2.rectangle(overlay, (TEXT_BOX_X, TEXT_BOX_Y), (TEXT_BOX_X + TEXT_BOX_WIDTH, TEXT_BOX_Y + TEXT_BOX_HEIGHT), text_bg_color, -1)
        
        # Image key
        cv2.putText(overlay, f"Image: {row['image_key']}", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        y_offset += TEXT_LINE_SPACING
        
        # Gaze error
        if not pd.isna(row['gaze_error']):
            cv2.putText(overlay, f"Gaze Error: {row['gaze_error']:.4f}", (TEXT_START_X, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        else:
            cv2.putText(overlay, "Gaze Error: N/A", (TEXT_START_X, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        y_offset += TEXT_LINE_SPACING
        
        # Angular gaze error
        if not pd.isna(row['angular_gaze_error']):
            cv2.putText(overlay, f"Angular Error: {row['angular_gaze_error']:.2f}", (TEXT_START_X, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        else:
            cv2.putText(overlay, "Angular Error: N/A", (TEXT_START_X, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        y_offset += TEXT_LINE_SPACING
        
        # Legend
        cv2.putText(overlay, "GT=Green, PRED=Red", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, LEGEND_FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        
        # Save the visualization
        output_filename = f"{row['image_key']}_gaze_comparison.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), overlay)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} images...")
    
    print(f"Visualization complete! Processed {processed_count} images.")
    print(f"Results saved to: {output_dir}")


def visualize_single_gaze_result(csv_path, output_dir, image_key=None, row_index=None):
    """
    Visualize a single gaze result by image key or row index
    
    Args:
        csv_path: Path to the gaze results CSV file
        output_dir: Directory to save visualization
        image_key: Image key to visualize (e.g., '00000001')
        row_index: Row index to visualize (0-based)
    """
    import pandas as pd
    import cv2
    import os
    import ast
    
    # UI Parameters - same as main function for consistency
    TEXT_BOX_WIDTH = 250
    TEXT_BOX_HEIGHT = 85
    TEXT_BOX_X = 10
    TEXT_BOX_Y = 5
    TEXT_START_X = 15
    TEXT_START_Y = 25
    TEXT_LINE_SPACING = 15
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    LEGEND_FONT_SCALE = 0.4
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Select the row
    if image_key is not None:
        row_mask = df['image_key'] == image_key
        if not row_mask.any():
            print(f"Image key '{image_key}' not found in dataset")
            return
        row = df[row_mask].iloc[0]
        row_idx = df[row_mask].index[0]
    elif row_index is not None:
        if row_index < 0 or row_index >= len(df):
            print(f"Row index {row_index} out of range (0-{len(df)-1})")
            return
        row = df.iloc[row_index]
        row_idx = row_index
    else:
        print("Please specify either image_key or row_index")
        return
    
    # Process the single row
    image_path = row['image_path']
    
    # Try to load image from various possible paths
    possible_paths = [
        image_path,
        os.path.join('/mnt/d/Projects/data/gazefollow', image_path),
        os.path.join('/mnt/d/Projects/data/gazefollow/train', image_path),
        os.path.join('/mnt/d/Projects/data/gazefollow/test', image_path),
    ]
    
    img = None
    for full_image_path in possible_paths:
        if os.path.exists(full_image_path):
            img = cv2.imread(full_image_path)
            if img is not None:
                break
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
        
    h, w, _ = img.shape
    
    # Create visualization overlay
    overlay = img.copy()
    
    # --- Ground Truth Visualization (Green) ---
    # GT Body bbox
    if not pd.isna(row['body_bbox_x']) and not pd.isna(row['body_bbox_y']):
        gt_x1 = int(row['body_bbox_x'] * w)
        gt_y1 = int(row['body_bbox_y'] * h)
        gt_x2 = int((row['body_bbox_x'] + row['body_bbox_width']) * w)
        gt_y2 = int((row['body_bbox_y'] + row['body_bbox_height']) * h)
        cv2.rectangle(overlay, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 3)  # Green
        cv2.putText(overlay, "GT", (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # GT Eye position (small circle)
    if not pd.isna(row['eye_x']) and not pd.isna(row['eye_y']):
        gt_eye_x = int(row['eye_x'] * w)
        gt_eye_y = int(row['eye_y'] * h)
        cv2.circle(overlay, (gt_eye_x, gt_eye_y), 5, (0, 255, 0), -1)  # Green dot
    
    # GT Gaze target (larger circle)
    if not pd.isna(row['gaze_x']) and not pd.isna(row['gaze_y']):
        gt_gaze_x = int(row['gaze_x'] * w)
        gt_gaze_y = int(row['gaze_y'] * h)
        cv2.circle(overlay, (gt_gaze_x, gt_gaze_y), 15, (0, 255, 0), 3)  # Green circle
        cv2.putText(overlay, "GT Gaze", (gt_gaze_x+20, gt_gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # --- Prediction Visualization (Red) ---
    # Predicted person bbox
    if not pd.isna(row['llava_person_bbox']):
        try:
            # Parse the bbox string (could be list format)
            bbox_str = str(row['llava_person_bbox']).strip()
            if bbox_str.startswith('[') and bbox_str.endswith(']'):
                pred_bbox = ast.literal_eval(bbox_str)
                if len(pred_bbox) == 4:
                    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
                    cv2.rectangle(overlay, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 3)  # Red
                    cv2.putText(overlay, "PRED", (pred_x1, pred_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except:
            pass
    
    # Predicted gaze points
    if not pd.isna(row['llava_gaze_points']):
        try:
            # Parse the gaze points string (numpy array format)
            gaze_str = str(row['llava_gaze_points']).strip()
            if gaze_str.startswith('[') and gaze_str.endswith(']'):
                # Remove brackets and clean up the string
                content = gaze_str[1:-1].strip()
                # Split by whitespace and filter out empty strings
                parts = [x for x in content.split() if x]
                if len(parts) >= 2:
                    pred_gaze_x = int(float(parts[0]))
                    pred_gaze_y = int(float(parts[1]))
                    cv2.circle(overlay, (pred_gaze_x, pred_gaze_y), 15, (0, 0, 255), 3)  # Red circle
                    cv2.putText(overlay, "PRED Gaze", (pred_gaze_x+20, pred_gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error parsing gaze points for {row['image_key']}: {e}")
    
    # --- Error Information (Top-left corner) ---
    y_offset = TEXT_START_Y
    
    # Background rectangle for text
    text_bg_color = (0, 0, 0)  # Black background
    cv2.rectangle(overlay, (TEXT_BOX_X, TEXT_BOX_Y), (TEXT_BOX_X + TEXT_BOX_WIDTH, TEXT_BOX_Y + TEXT_BOX_HEIGHT), text_bg_color, -1)
    
    # Image key
    cv2.putText(overlay, f"Image: {row['image_key']}", (TEXT_START_X, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    y_offset += TEXT_LINE_SPACING
    
    # Gaze error
    if not pd.isna(row['gaze_error']):
        cv2.putText(overlay, f"Gaze Error: {row['gaze_error']:.4f}", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    else:
        cv2.putText(overlay, "Gaze Error: N/A", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    y_offset += TEXT_LINE_SPACING
    
    # Angular gaze error
    if not pd.isna(row['angular_gaze_error']):
        cv2.putText(overlay, f"Angular Error: {row['angular_gaze_error']:.2f}", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    else:
        cv2.putText(overlay, "Angular Error: N/A", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    y_offset += TEXT_LINE_SPACING
    
    # Person ID
    if not pd.isna(row['llava_matched_person_id']):
        cv2.putText(overlay, f"Person ID: {int(row['llava_matched_person_id'])}", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    else:
        cv2.putText(overlay, "Person ID: N/A", (TEXT_START_X, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    
    # Save the image
    output_filename = f"{row['image_key']}_gaze_comparison.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, overlay)
    
    print(f"Visualization saved: {output_path}")
    return output_path
