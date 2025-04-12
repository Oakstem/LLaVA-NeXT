import os
print(os.getcwd())
from operator import attrgetter
from pathlib import Path

import pandas as pd

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from docs.clip_eval import fix_wsl_paths, save_cls
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import subprocess

result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Print the output
print(result.stdout)
slurm = os.environ.get("SLURM_JOB_ID", None)
print(f"SLURM_JOB_ID: {slurm}")
prefix = None
def get_best_gpu():
    best_gpu = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.mem_get_info(i)[0]  # free memory
    )
    device = torch.device(f"cuda:{best_gpu}")
    print(f"Best GPU: {best_gpu}")
    return device

def get_hostname():
    import socket
    hostname = socket.gethostname()
    print("Running on:", hostname)
    return hostname
hostname = get_hostname()

if torch.cuda.is_available():
    print(f"PyTorch cuda version:{torch.version.cuda}")
    print(torch.cuda.get_device_name(0))
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")

if 'psychology' in hostname:
    prefix = "/home/new_storage/"
    # os.environ["HF_HOME"] = "/home/new_storage/HuggingFace_cache"
    # os.environ["TRANSFORMERS_CACHE"] = "/home/new_storage/HuggingFace_cache"
    # os.environ["HF_DATASETS_CACHE"] = "/home/new_storage/HuggingFace_cache"
    # os.environ["HF_TOKENIZERS_CACHE"] = "/home/new_storage/HuggingFace_cache"
elif slurm:
    # slurm = os.environ.get("SLURM_JOB_ID")
    prefix = "/home/ai_center/ai_users/alonmardi/"
if prefix is not None:
    print(f"Setting HF paths prefix to {prefix}")
    os.environ["HF_HOME"] = os.path.join(prefix, "HuggingFace_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(prefix, "HuggingFace_cache")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(prefix, "HuggingFace_cache")
    os.environ["HF_TOKENIZERS_CACHE"] = os.path.join(prefix, "HuggingFace_cache")

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
model_name = "llava_qwen"
# device = "cuda"
device = get_best_gpu()
print(f"Using device: {device}")
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video
def load_movie(video_path, max_frames_num, output_dir, vr=None, start_frame=0, fps=25, duration=3, **kwargs):
    if vr is None:
        vr = VideoReader(video_path, ctx=cpu(0))
    end_frame = start_frame + fps * duration
    uniform_sampled_frames = np.linspace(start_frame, end_frame, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    res = {'frames': spare_frames, 'start_frame': start_frame, 'end_frame': end_frame, 'vr': vr}
    # save the video in the output directory
    output_path = os.path.join(output_dir, f"{start_frame}_{end_frame}.mp4")
    # create_video_clip(spare_frames, output_path)

    return res

# create and save compressed video clip
def create_video_clip(frames, output_path):
    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 25, (frames.shape[2], frames.shape[1]))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video clip saved at {output_path}")

    return output_path


# Load and process video
if 'psychology' in hostname:
    # png_dir = r"/home/new_storage/experiments/NND/500daysofsummer/png"
    # annot_df_path = "/home/Alon/data/summer/annotations/frame_distances_avg_1s_full.csv"
    video_path = r"/home/new_storage/experiments/NND/500 Days Of Summer.2009.720p.BDRip.x264-VLiS.mp4"
elif slurm:
    video_path = "/home/ai_center/ai_data/alonmardi/Sherlock.S01E01.A.Study.in.Pink.mkv"
else:
    video_path = r"D:\Projects\Annotators\data\Sherlock.S01E01.A.Study.in.Pink.mkv"
    # video_path = r"E:\moments\Moments_in_Time_layla\training\working\t2z1X76v4ys_178.mp4"
    video_path = video_path.replace("\\", os.sep)
    drive = video_path.split(os.sep)[0]
    video_path = video_path.replace(drive, f'/mnt/{drive[0].lower()}')
output_dir = Path(video_path).parent / 'Sherlock_facing_10s'
output_dir.mkdir(exist_ok=True, parents=True)
result_path = output_dir / 'llava_3s_video_results_facing_10s.csv'
result_df = pd.DataFrame(columns=['seconds', 'start_frame', 'end_frame', 'question', 'response'])

offset = 0
duration = 10
avg_pool_img_embeds = True
vr = VideoReader(video_path, ctx=cpu(0))
fps = vr.get_avg_fps()
step = int(fps * duration)
step = int(fps * 1.5)   # with overlap
# offset = 0

for frame_ind in range(offset, len(vr), step):
    video_frames_dd = {"frames": None, "start_frame": frame_ind, "vr": vr, "max_frames_num": 16,
                       "video_path": video_path, "fps": fps, "duration": duration, "output_dir": output_dir}
    video_frames_dd = load_movie(**video_frames_dd)
    video_frames = video_frames_dd['frames']

    # print(video_frames.shape) # (16, 1024, 576, 3)
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    # Prepare conversation input
    conv_template = "qwen_1_5"
    # joint action
    # question = (f"Describe what each person is doing, and conclude whether they are engaged in a joint action.")
    # communication
    question = (f"Describe the scene and interactions, then determine if the people are communicating and how")
    # general social description
    question = (f"Describe where each of people is located in the video, what are they doing, and where are they looking")
    # facing
    question = (f"Describe where each person is looking, and conclude whether they are facing each other.")
    full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n {question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont, image_embeds = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_hidden_states=True,
        return_dict_in_generate=True,
        modalities=["video"],
    )
    image_embeds = image_embeds[0]
    if avg_pool_img_embeds:
        image_embeds = torch.mean(image_embeds, dim=0)

    text_outputs = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)[0]
    frame_time = frame_ind // fps
    # convert to minutes : seconds
    minutes = frame_time // 60
    seconds = frame_time % 60
    print(f"Frame: {frame_time} ({minutes}:{seconds})")
    print(text_outputs)
    language_latent = cont.hidden_states[-1][-1].view(-1)
    save_cls(str(frame_ind), language_latent, output_dir=output_dir / 'llm_language_embeds')
    save_cls(str(frame_ind), image_embeds, output_dir=output_dir / 'vision_only_embeds')

    clip_res = {'start_frame': video_frames_dd['start_frame'], 'end_frame': video_frames_dd['end_frame'],
                'seconds': frame_ind // fps,  'question': question, 'response': text_outputs}
    clip_df = pd.DataFrame(clip_res, index=[0])
    result_df = pd.concat([result_df, clip_df], ignore_index=True)
    # save every 10 responses
    if len(result_df) % 10 == 0:
        result_df.to_csv(result_path, index=False)
