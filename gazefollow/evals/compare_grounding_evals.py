import json
import pandas as pd

res1_path = "evaluation_results/eval_20251024_215845/evaluation__run_model_generation_results_20251029_170525/dataset_localization_results.json"
res2_path = "evaluation_results/baseline_llava/dataset_localization_results.json"

with open(res1_path, "r") as f:
    data1 = json.load(f)

with open(res2_path, "r") as f:
    data2 = json.load(f)

def find_errored_samples(data):
    errored_ids = set()
    for entry in data:
        if entry.get("error"):
            errored_ids.add(entry.get("id"))
    return errored_ids

# lets flatten the data into a pandas dataframe for easier analysis
df1 = pd.json_normalize(data1)
df2 = pd.json_normalize(data2)

# missing gaze detections
df1_missing_target_ground = df1.loc[df1['gaze_detections.person_1.gaze_coordinates'].isna()]
df2_missing_target_ground = df2.loc[df2['gaze_detections.person_1.gaze_coordinates'].isna()]

print(f"Model 1 missing gaze grounding for {len(df1_missing_target_ground)} samples")
print(f"Model 2 missing gaze grounding for {len(df2_missing_target_ground)} samples")

pass