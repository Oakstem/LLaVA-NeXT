import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from docs.research_utils import fix_wsl_paths
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import torch
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

from docs.research_utils import fix_wsl_paths

task_prompt = '<REGION_TO_DESCRIPTION>'
# annot_path = r"D:\Projects\data\gazefollow\test_annotations_release.txt"
annot_path = r"D:\Projects\data\gazefollow\train_annotations_release.txt"
split_type = "train" if "train" in annot_path else "test2"
base_data_dir_path = fr"D:\Projects\data\gazefollow"
llava_results_dir = r"D:\Projects\LLaVA-NeXT\llava_attention_sweep\20250503_001255_You_are_an_expert_vision_assis"
llava_results_dir = Path(fix_wsl_paths(llava_results_dir))
base_data_dir_path = Path(fix_wsl_paths(base_data_dir_path))
annot_path = fix_wsl_paths(annot_path)


def load_gt_data(annot_path):
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

    # Convert all the numerical columns to numeric types
    numeric_columns = ['id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                       'eye_x', 'eye_y', 'gaze_x', 'gaze_y',
                       'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=8,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def group_by_image(df):
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
    return compact_df

# main
if __name__ == "__main__":
    df = load_gt_data(annot_path)
    compact_df = group_by_image(df)

    df['person_desc'] = ''
    save_interval = 5
    results_dd = {}
    # iterate and extract person descriptions for each image
    progr = tqdm(compact_df.iterrows(), total=compact_df.shape[0])
    for index, row in progr:
        # get the relevant rows in the original dataframe
        relevant_rows = df['image_path'] == row['image_path']
       # load the image
        img_path = base_data_dir_path / row['image_path']
        if not img_path.exists():
           continue
        img = Image.open(img_path).convert("RGB")
        h = 1   #img.height
        w = 1   #img.width
        gt_person_bbox = [w * row['body_bbox_x'], h * row['body_bbox_y'],
                         w * (row['body_bbox_x'] + row['body_bbox_width']),
                         h * (row['body_bbox_y'] + row['body_bbox_height'])]

        # get the gaze target
        gaze_target_x = int(gt_person_bbox[0] * 1000)
        gaze_target_y = int(gt_person_bbox[1] * 1000)
        gaze_target_x_end = int(gt_person_bbox[2] * 1000)
        gaze_target_y_end = int(gt_person_bbox[3] * 1000)

        results = run_example(task_prompt, img, text_input=f"<loc_{gaze_target_x}><loc_{gaze_target_y}>"
                                                          f"<loc_{gaze_target_x_end}><loc_{gaze_target_y_end}>")
        desc = results.get(task_prompt,'').split('<')[0]
        progr.set_description(f"Person description:{desc}")
        # save the person description in the dataframe
        # df.loc[relevant_rows, 'person_desc'] = desc
        results_dd[row['image_path']] = desc

        # save the results
        if index % save_interval == 0:
            # save the results to a json file
            results_file = Path(annot_path).parent / f"{split_type}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_dd, f, indent=4)
            # save the results to a csv file
             # df.to_csv(llava_results_dir / f"{split_type}_results.csv", index=False)



