from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_ANNOTATIONS = "datasets/Vacation/test_annotations.csv"
DEFAULT_FRAMES_DIR = "datasets/Vacation/frames"
VACATION_LABELS = [
    "MutualGaze",
    "SharedObjectAttention",
    "OneSidedGaze",
    "NonCommmunicative",
    "None",
]


def build_frame_path(frames_dir: Path, video_id: int, frame_id: int) -> Path:
    return frames_dir / str(video_id) / f"{frame_id + 1:06d}.png"


def select_frames_with_skipping(df: pd.DataFrame, skip_step: int) -> pd.DataFrame:
    if skip_step <= 1:
        return (
            df.groupby(["video_id", "frame_id"])
            .agg({"event_attribute": "first"})
            .reset_index()
        )

    unique_frames = (
        df.groupby(["video_id", "frame_id"])
        .agg({"event_attribute": "first"})
        .reset_index()
        .sort_values(["video_id", "frame_id"])
    )

    selected_frames = []
    for video_id in unique_frames["video_id"].unique():
        video_frames = unique_frames[unique_frames["video_id"] == video_id].copy()
        video_frames = video_frames.sort_values("frame_id").reset_index(drop=True)
        video_frames["event_changed"] = (
            video_frames["event_attribute"] != video_frames["event_attribute"].shift(1)
        ) | (video_frames["frame_id"] != video_frames["frame_id"].shift(1) + 1)
        video_frames["run_id"] = video_frames["event_changed"].cumsum()

        for run_id in video_frames["run_id"].unique():
            run_frames = video_frames[video_frames["run_id"] == run_id]
            sampled = run_frames.iloc[::skip_step]
            selected_frames.append(sampled[["video_id", "frame_id", "event_attribute"]])

    return pd.concat(selected_frames, ignore_index=True)


def randomize_frame_queue(
    selected_frames: pd.DataFrame, randomize: bool, seed: int | None
) -> pd.DataFrame:
    if not randomize:
        return selected_frames.reset_index(drop=True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return selected_frames.sample(frac=1, random_state=seed).reset_index(drop=True)
