import pandas as pd

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
