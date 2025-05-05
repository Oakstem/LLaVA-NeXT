import os
from operator import attrgetter
from pathlib import Path
import socket
import warnings
import glob
import copy
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd
import torch
import cv2
import numpy as np
from PIL import Image
import requests
from decord import VideoReader, cpu
from transformers import PreTrainedModel, PreTrainedTokenizer
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_environment(hostname: str) -> None:
    """Setup environment variables based on hostname."""
    if 'psychology' in hostname:
        cache_dir = "/home/new_storage/HuggingFace_cache"
        os.environ.update({
            "HF_HOME": cache_dir,
            "TRANSFORMERS_CACHE": cache_dir,
            "HF_DATASETS_CACHE": cache_dir,
            "HF_TOKENIZERS_CACHE": cache_dir
        })

def get_hostname() -> str:
    """Get the current hostname."""
    hostname = socket.gethostname()
    print("Running on:", hostname)
    return hostname

def fix_wsl_paths(path: str) -> str:
    """Convert Windows paths to WSL paths if necessary."""
    # If path already starts with /mnt/, assume it's correct WSL format
    if path.startswith('/mnt/'):
        return path
    # Otherwise, attempt conversion from Windows format
    path = path.replace("\\", os.sep)
    drive_parts = path.split(os.sep)
    if len(drive_parts) > 0 and len(drive_parts[0]) > 1 and drive_parts[0][1] == ':':
        drive_letter = drive_parts[0][0].lower()
        # Reconstruct path starting from /mnt/<drive_letter>/...
        wsl_path = f'/mnt/{drive_letter}/' + os.sep.join(drive_parts[1:])
        return wsl_path
    else:
        # If it doesn't look like a Windows path either, return original
        return path

def load_model() -> Tuple[PreTrainedTokenizer, PreTrainedModel, SigLipImageProcessor, int]:
    """Load and initialize the LLaVA model."""
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
    model_name = "llava_qwen"
    device_map = "auto"
    llava_model_args = {"multimodal": True}

    return load_pretrained_model(
        pretrained, None, model_name,
        device_map=device_map,
        attn_implementation="sdpa",
        **llava_model_args
    )

def load_video(
    video_path: Union[str, Path],
    max_frames_num: int,
    output_dir: Union[str, Path],
    vr: Optional[VideoReader] = None,
    start_frame: int = 0,
    fps: int = 25,
    duration: int = 3,
    **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Load and process video frames."""
    if vr is None:
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    end_frame = start_frame + len(vr) - 1
    uniform_sampled_frames = np.linspace(start_frame, end_frame, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return {
        'frames': spare_frames,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'vr': vr
    }

def save_cls(image_path: Union[str, Path], cls: torch.Tensor, output_dir: Union[str, Path] = "llm_image_embeds") -> None:
    """Save classifier tensor to file."""
    save_path = Path(output_dir)
    save_path.mkdir(exist_ok=True)
    cls_path = save_path / Path(image_path).with_suffix(".pt").name
    torch.save(cls, cls_path)

def process_single_video(
    video_path: Union[str, Path],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    image_processor: SigLipImageProcessor,
    output_dir: Union[str, Path],
    fps: int = 25,
    **kwargs: Any
) -> Optional[Dict[str, str]]:
    """Process a single video through the model."""
    video_frames_dd = {
        "frames": None,
        "start_frame": 0,
        "vr": None,
        "max_frames_num": 16,
        "video_path": video_path,
        "fps": fps,
        "duration": 3,
        "output_dir": output_dir
    }

    video_frames_dd = load_video(**video_frames_dd)
    if video_frames_dd is None:
        return None

    video_frames = video_frames_dd['frames']
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()

    # Prepare conversation input
    conv_template = "qwen_1_5"
    question = "Describe the scene and interactions, then determine if the people are communicating and how."
    full_prompt = f"{DEFAULT_IMAGE_TOKEN}\n {question}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont, image_embeds = model.generate(
        input_ids,
        images=[frames],
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_hidden_states=True,
        return_dict_in_generate=True,
        modalities=["video"],
    )

    language_latent = cont.hidden_states[-1][-1].view(-1)
    save_cls(video_path, language_latent, output_dir=output_dir / 'llm_language_embeds')

    text_output = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)[0]
    print(text_output)

    return {
        'question': question,
        'response': text_output
    }

def find_video_paths(dataset_path: Union[str, Path]) -> Dict[str, str]:
    """Create a mapping of video names to their full paths."""
    video_paths = {}
    for video_file in glob.glob(os.path.join(str(dataset_path), "**/*.mp4"), recursive=True):
        video_name = os.path.basename(video_file)
        video_paths[video_name] = video_file
    return video_paths

def csv_routine(
    csv_path: Union[str, Path],
    dataset_path: Union[str, Path],
    result_df: pd.DataFrame,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    image_processor: SigLipImageProcessor,
    result_path: Union[str, Path],
    output_dir: Union[str, Path]
) -> None:
    """Process videos listed in a CSV file."""
    csv_df = pd.read_csv(csv_path)
    video_list = csv_df['video_name'].values
    video_paths = find_video_paths(dataset_path)

    for video in video_list:
        if video in result_df['video_id'].values:
            continue

        if video not in video_paths:
            print(f"Could not find video: {video}")
            continue

        result = process_single_video(
            video_paths[video], model, tokenizer,
            image_processor, output_dir
        )

        if result:
            clip_res = {'video_id': video, **result}
            clip_df = pd.DataFrame(clip_res, index=[0])
            result_df = pd.concat([result_df, clip_df], ignore_index=True)
            result_df.to_csv(str(result_path), index=False)

def video_list_routine(
    video_list: np.ndarray,
    dataset_path: Union[str, Path],
    result_df: pd.DataFrame,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    image_processor: SigLipImageProcessor,
    result_path: Union[str, Path],
    output_dir: Union[str, Path]
) -> None:
    """Process a list of videos."""
    for video in video_list:
        if video in result_df['video_id'].values:
            continue

        video_path = os.path.join(str(dataset_path), video)
        result = process_single_video(
            video_path, model, tokenizer,
            image_processor, output_dir
        )

        if result:
            clip_res = {'video_id': video, **result}
            clip_df = pd.DataFrame(clip_res, index=[0])
            result_df = pd.concat([result_df, clip_df], ignore_index=True)
            result_df.to_csv(str(result_path), index=False)

def main() -> None:
    """Main execution function."""
    # Setup environment
    hostname = get_hostname()
    setup_environment(hostname)

    # Initialize paths based on hostname
    if 'psychology' in hostname:
        dataset_path = r"/home/new_storage/experiments/Moments_In_Time/Moments_in_Time_layla"
        filt_csv_path = r"/home/new_storage/experiments/Moments_In_Time/llava_3s_moments_results_comm.csv"
        output_dir = Path(dataset_path).parent / 'llm_results'
        csv_path = Path(dataset_path).parent / 'llava_3s_moments_results_primitives2_w_gt.csv'
    else:
        dataset_path = fix_wsl_paths(r"E:\moments\Moments_in_Time_Raw\training")
        output_dir = fix_wsl_paths(r"D:\Projects\Annotators\data\moments")
        filt_csv_path = fix_wsl_paths(r"D:\Projects\Annotators\data\moments\llava_3s_moments_results_people_facing.csv")
        csv_path = fix_wsl_paths(output_dir / 'llava_3s_moments_results_primitives2_w_gt.csv')

    # Setup directories and load model
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    result_path = output_dir / 'llava_3s_moments_results_comm.csv'

    tokenizer, model, image_processor, max_length = load_model()
    model.eval()

    # Initialize or load results DataFrame
    result_df = pd.DataFrame(columns=['video_id', 'question', 'response'])
    if result_path.exists():
        result_df = pd.read_csv(result_path)

    # Process videos based on mode
    oswalk_dir = False
    use_csv = True

    if use_csv:
        csv_routine(csv_path, dataset_path, result_df, model, tokenizer, image_processor, result_path, output_dir)
    elif os.path.exists(filt_csv_path):
        filt_df = pd.read_csv(filt_csv_path)
        video_list = filt_df['full_video_id'].values
        video_list_routine(video_list, dataset_path, result_df, model, tokenizer, image_processor, result_path, output_dir)

if __name__ == "__main__":
    main()
