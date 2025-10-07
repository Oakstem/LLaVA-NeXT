%% Preparing GT data for evaluation 
To Evaluate our LLava ability to localize gaze correctly we need to do an inference with the original LLava model and ask it to identify where each person is looking.
For every person described, we need to find the one that is the closest to the Gaze origin in GT
Then do the same for the described gaze target (but this time without GT, just localize based on attention maps the point the model is describing)

To achieve this, we do the following steps:
1. `docs/llava_extract_ppl_loc_and_gaze.py`  
Do an inference with LLava on the gazefollow dataset with the following prompt:
"You are an expert vision assistant. 
Step 1 – Caption • Provide **one concise sentence** that broadly describes the entire scene. • Begin the line with:  Caption: Step 2 – Foreground people & gaze 
    1. Detect every person whose height ≥ 5 % of the image (foreground). 
    2. List them **left‑to‑right**. Number sequentially starting at 1. For each person output exactly **one line** in this format: Person {N}: {short description}, looking at {target | “outside the frame” | “uncertain”} Output format (no extra lines, no prose other than what is specified): 
    ------------------------------------------------- 
    Caption: {your one‑sentence scene description} 
    Person 1: {short description}, looking at … 
    Person 2: {short description}, looking at … … 
    ------------------------------------------------- 
Additional rules 
• Keep the phrase **“looking at”** unchanged. 
• {short description} = ≤ 6 words (e.g., “man in red jacket”). 
• If no foreground person is detected, write exactly: "No foreground people detected."
• If gaze cannot be determined, use “uncertain”. 
• Do **not** output your reasoning or any extra text" 

Example result: 
Caption: Former President Barack Obama claps and smiles as he stands at a podium, surrounded by a crowd of people who are also clapping.

Person 1: Former President Barack Obama, clapping and smiling, looking at the crowd.
Person 2: A man in a dark suit, clapping, looking at the crowd.
Person 3: A woman with long hair, clapping, looking at the crowd.
Person 4: A man in a light-colored shirt, clapping, looking at the crowd.
Person 5: A man in a dark suit, clapping, looking at the crowd.

For every image, we save the generated text + attention map on the image coordinates of every generated word. 

2. [old single core script]`gazefollow/temporal_attn_focus_refactored.py`  
[new parallel proc script] `python gazefollow/launch_parallel.py "/mnt/d/Projects/data/gazefollow/results/valid_runs/20250929_195104_You_are_an_expert_vision_assis" "/mnt/d/Projects/data/gazefollow/train" --config default --workers 8`
Based on the given attention maps, create centered locations for every person & and it's gaze target
3. To generate the Gaze source&target masks, do:
In Grounded_SAM repo, run:  
`/home/alonz/gd_sam/bin/python /mnt/d/Projects/Grounded-SAM-2/process_auto.py --attn-base-dir "D:\Projects\data\gazefollow\results\valid_runs\20250902_015248_You_are_an_expert_vision_assis" --images-base-dir "D:\Projects\data\gazefollow\train"`
This generates segmentation masks for every described person + it's target gaze based on the centered locations from the previous step
4. `finetune/gaze_follow_ds_refactored.py`  
Final script that goes over all the results, finds the actual person that is closest to the GT location, and check's it's error from the GT gaze target location.
We're using 2 metrics to define the error:
    1. Normalized l2 distance
    2. 2D angular distance

%% Conversation dataset preparing
1. `build_json_ds.py` - combines both the 'baseline' llava run with full descriptions (using only people description here) and the steered attention llava results, saves the result to `combined_description_results.csv`  
2. `create_sgl_conversations.py` - builds a conversation json dataset from the results, prefers original llava person descriptions and adds the generated target descriptions
3. `fix_duplicated_words.py` - fixing and filtering the resulted json by removing duplicate words, ('the the'), 'man1/man2' words etc. Also filtering images with less than set threshold of people.
4. `split_dataset.py` - split the resulted json to train and val. 


## Running the train script
`cd /mnt/d/Projects/LLaVA-NeXT && export PYTHONPATH="${PYTHONPATH}:/mnt/d/Projects/LLaVA-NeXT" && ~/llava/bin/python llava/train/train.py   --model_name_or_path "lmms-lab/llava-onevision-qwen2-7b-ov-chat"   --data_path "/mnt/d/Projects/data/gazefollow/results/valid_runs/sgl_conversation_data.json"   --output_dir "/mnt/d/Projects/LLaVA-NeXT/training_outputs"   --lora_enable True   --lora_r 4   --lora_alpha 16   --lora_target_modules "lm_head"   --num_train_epochs 1   --per_device_train_batch_size 1   --save_steps 100   --logging_steps 10 --image_folder "/mnt/d/Projects/data/gazefollow"`