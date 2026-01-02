import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import shutil

def parse_vacation_test_frames():
    # Hardcoded test list from readme.txt
    test_list = ['217','230', '232', '233', '236', '46', '58', '62', '63', '64', '65', '7', '71',
                 '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85',
                 '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

    base_path = 'datasets/Vacation'
    annotation_path = os.path.join(base_path, 'annotation_cleaned')
    video_path = os.path.join(base_path, 'Videos')
    frames_base_path = os.path.join(base_path, 'frames')
    csv_output_path = os.path.join('vacation', 'test_annotations.csv')

    if not os.path.exists(frames_base_path):
        os.makedirs(frames_base_path)

    print(f"Processing {len(test_list)} videos from the test split...")
    
    all_annotations = []

    for video_id in tqdm(test_list):
        output_dir = os.path.join(frames_base_path, video_id)
        
        # Cleanup existing directory or create new
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 1. Parse Annotation
        ant_file = os.path.join(annotation_path, f"NewAnt_{video_id}.txt")
        annotated_frames = set()
        bbx_count = 0
        if os.path.exists(ant_file):
            with open(ant_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 13: # Ensure we have enough parts based on readme
                    try:
                        # bbx ID | xmin | ymin| xmax | ymax | frame ID | lost | occluded | generated | bbx label | event attribute | atomic attribute | attention focus
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
                            "event_attribute": parts[10],
                            "atomic_attribute": parts[11],
                            "attention_focus": parts[12]
                        }
                        
                        all_annotations.append(ant_data)
                        annotated_frames.add(frame_id)
                        bbx_count += 1
                    except ValueError:
                        pass
        else:
             print(f"Warning: Annotation file not found for {video_id}")

        # 2. Extract Frames
        vid_file = os.path.join(video_path, f"{video_id}.mp4")
        if not os.path.exists(vid_file):
            print(f"Warning: Video file not found for {video_id}")
            continue

        cap = cv2.VideoCapture(vid_file)
        frame_idx = 1 
        
        extracted_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as PNG
            save_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(save_path, frame)
            extracted_count += 1
            frame_idx += 1
        
        cap.release()

        # Summary for this video
        print(f"Video {video_id}: Extracted {extracted_count} frames. Annotations: {bbx_count} boxes on {len(annotated_frames)} frames.")

    # Save aggregated annotations
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        df.to_csv(csv_output_path, index=False)
        print(f"Saved {len(df)} annotations to {csv_output_path}")
    else:
        print("No annotations found.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    parse_vacation_test_frames()
