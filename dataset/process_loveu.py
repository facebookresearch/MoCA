# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import os
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
import argparse
import imageio
import ffmpeg
import pandas as pd
import tempfile


OBJECT = "object"
STYLE = "style"
MOTION = "motion"
BACKGROUND = "background"
MULTI_SPATIAL = "multi_spatial"
MULTI_MOTION = "multi_motion"

NAMES = set([
    'gold-fish', 'trucks-race', 'varanus-cage', 'squirrel-climb', 'dirt-road-driving', 
    'audi-snow-trail', 'mallard-duck-flight', 'eiffel-flyover', 'las-vegas-time-lapse', 
    'warsaw-multimedia-fountain', 'geometric-video-background', 
    'typewriter-super-slow-motion', 'raindrops', 'lotus', 'earth-full-view', 
    'setting-sun', 'cat-in-the-sun', 'swans', 'red-roses-sunny-day', 'singapore-airbus-a380-landing', 
    'fireworks-display', 'seagull-flying', 'aircraft-landing', 'sharks-swimming', 
    'bird-on-feeder', 'cows-grazing', 'ferris-wheel-timelapse', 'butterfly-feeding-slow-motion', 
    'ski-lift-time-lapse', 'ship-sailing', 'deer-eating-leaves', 'airplane-and-contrail', 
    'wind-turbines-at-dusk', 'american-flag-in-wind', 'pouring-beer-from-bottle'
])


def read_video(input_file, fps=4, resolution=256, T=2, offset_W=0, offset_T=0):
    input_file = str(input_file)
    process = (
        ffmpeg.input(input_file)
        .filter('fps', fps=fps, round='up')
        .filter('scale', w=resolution, h=resolution, force_original_aspect_ratio='increase')
        .filter('crop', resolution, resolution)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    video, _ = process.communicate()
    video = np.frombuffer(video, np.uint8).reshape(-1, resolution, resolution, 3)
    video = video[int(offset_T * fps):int(offset_T*fps)+T*args.fps]
    assert video.shape == (T*args.fps, resolution, resolution, 3), video.shape
    return video


def main():
    data_dir = Path(args.output_folder)
    data_dir.mkdir(parents=True, exist_ok=True)

    loveu_dir = Path(args.loveu_folder)
    csv_file = loveu_dir / "LOVEU-TGVE-2023_Dataset.csv"
    df = pd.read_csv(csv_file)

    davis_idx = df.index[df['Video name'] == 'DAVIS Videos:'].tolist()[0]
    youtube_idx = df.index[df['Video name'] == 'Youtube Videos:'].tolist()[0]
    videvo_idx = df.index[df['Video name'] == 'Videvo Videos:'].tolist()[0]

    edit_type_mapping = {}
    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        if davis_idx < row_idx < youtube_idx - 1:
            video_folder = "DAVIS_480p"
        elif youtube_idx < row_idx < videvo_idx - 1:
            video_folder = 'youtube_480p'
        elif row_idx > videvo_idx:
            video_folder = 'videvo_480p'
        else:
            continue
        
        if row['Video name'] not in NAMES:
            continue
            
        video_path = loveu_dir / f"{video_folder}/480p_videos/{row['Video name']}.mp4"
        video = read_video(video_path, fps=args.fps, resolution=args.resolution)
        caption = row["Our GT caption"]
        keys = ["Style Change Caption", "Object Change Caption", "Background Change Caption", "Multiple Changes Caption"]
        edit_types = [STYLE, OBJECT, BACKGROUND, MULTI_SPATIAL]
        for j, k in enumerate(keys):
            edit_type_mapping[f"{row['Video name']}_{j:02}"] = edit_types[j]
            imageio.mimsave(data_dir / f"{row['Video name']}_{j:02}.mp4", video, fps=args.fps)
            with (data_dir / f"{row['Video name']}_{j:02}.txt").open('w') as f:
                f.write(f"{caption}\n{row[k]}")

    with (data_dir / 'edit_type_map.json').open('w') as f:
        json.dump(edit_type_mapping, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--loveu_folder", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()
    main()
