import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import shutil

def parse_vacation_test_frames_complete():
    # Hardcoded test list from readme.txt
    test_list = ['217','230', '232', '233', '236', '46', '58', '62', '63', '64', '65', '7', '71',
                 '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85',
                 '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

    base_path = 'datasets/Vacation'
    annotation_path = os.path.join(base_path, 'annotation_cleaned')
    csv_output_path = os.path.join('vacation', 'test_annotations_complete.csv')

    print(f"Processing {len(test_list)} videos from the test split...")
    
    all_annotations = []

    for video_id in tqdm(test_list):
        # 1. Parse Annotation
        ant_file = os.path.join(annotation_path, f"NewAnt_{video_id}.txt")
        if os.path.exists(ant_file):
            with open(ant_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                # Minimal requirement: down to bbx_label (index 9) -> 10 columns
                if len(parts) >= 10: 
                    try:
                        # bbx ID | xmin | ymin| xmax | ymax | frame ID | lost | occluded | generated | bbx label | ...
                        frame_id = int(parts[5].strip())
                        
                        ant_data = {
                            "video_id": video_id,
                            "bbx_id": parts[0],
                            "xmin": int(parts[1]),
                            "ymin": int(parts[2]),
                            "xmax": int(parts[3]),
                            "ymax": int(parts[4]),
                            "frame_id": frame_id,
                            "lost": parts[6],
                            "occluded": parts[7],
                            "generated": parts[8],
                            "bbx_label": parts[9],
                        }
                        
                        # Optional columns
                        if len(parts) >= 13:
                            ant_data["event_attribute"] = parts[10]
                            ant_data["atomic_attribute"] = parts[11]
                            ant_data["attention_focus"] = parts[12]
                        else:
                             ant_data["event_attribute"] = None
                             ant_data["atomic_attribute"] = None
                             ant_data["attention_focus"] = None
                        
                        all_annotations.append(ant_data)
                    except ValueError:
                        pass
        else:
             print(f"Warning: Annotation file not found for {video_id}")

    # Save aggregated annotations
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        df.to_csv(csv_output_path, index=False)
        print(f"Saved {len(df)} annotations to {csv_output_path}")
    else:
        print("No annotations found.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    parse_vacation_test_frames_complete()
