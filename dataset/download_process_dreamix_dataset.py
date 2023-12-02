# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import tempfile
import json
from pathlib import Path
import requests

import imageio
import numpy as np
from tqdm import tqdm
import ffmpeg

OBJECT = "object"
STYLE = "style"
MOTION = "motion"
BACKGROUND = "background"
MULTI_SPATIAL = "multi_spatial"
MULTI_MOTION = "multi_motion"


data = [
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_cake.mp4", 1, 0.3, 2, "A knife is cutting a papaya on a red plate", 
        ["A knife is cutting a cake on a red plate"], [OBJECT],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_swans.mp4", 1, 0.0, 1, "A beach with palm trees and water", 
        ["A beach with palm tree and swans in the water"], [OBJECT],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_circle.mp4", 1, 0.3, 0, "A hand writing on a paper", 
        ["A hand drawing a big circle on a paper", "A robot claw writing on a paper"], 
        [MULTI_MOTION, OBJECT],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/banner_video.mp4", 0, 0.0, 0, "A monkey eating food", 
        ["A bear dancing and jumping to upbeat music, moving his whole body"], 
        [MULTI_MOTION],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_leaping.mp4", 1,  0.15, 0, "A puppy walking", 
        ["A puppy leaping", "A puppy walking with a party hat"], 
        [MOTION, OBJECT],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_truck_river.mp4", 1, 0.4, 0, "Walking around an old pickup truck", 
        ["Zooming out from an old pickup truck", "An old pickup truck carrying wood logs", "An old pickup truck crossing a deep river"],
        [MOTION, OBJECT, MULTI_MOTION],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_saxophone.mp4", 1, 0.22, 0, "A man playing a saxophone", 
        ["A man playing a saxophone with musical notes flying out"], [STYLE],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_skateboard.mp4", 1,  0.15, 0, "A deer walking in a forest", 
        ["A deer rolling on a skateboard in a forest"], [OBJECT],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_fire.mp4", 1, 0.25, 0, "Walking through a field on a wooden path", 
        ["Walking through a field on a wooden path with fire on all sides"], [BACKGROUND],
    ),
    (
        "https://dreamix-video-editing.github.io/static/videos/vid2vid_noodles.mp4", 1, 0.25, 0, "stirring onions in a pot", 
        ["stirring noodles in a pot"], [OBJECT],
    ),
]


def read_process_video(input_file, fps=4, resolution=256, T=2, offset_W=0, offset_T=0):
    process = (
        ffmpeg.input(input_file)
        .filter('fps', fps=fps, round='up')
        .filter('scale', w=resolution, h=resolution, force_original_aspect_ratio='increase')
        .filter('crop', resolution, resolution, f"{offset_W}*iw", 0)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    video, _ = process.communicate()
    video = np.frombuffer(video, np.uint8).reshape(-1, resolution, resolution, 3)
    video = video[int(offset_T * fps):int(offset_T*fps)+T*fps]
    assert video.shape == (T*fps, resolution, resolution, 3), video.shape
    return video


def write_video(output_file, video, fps=30):
    output_writer = imageio.get_writer(output_file, fps=fps)
    for frame in video:
        output_writer.append_data(frame)
    output_writer.close()


def main():
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    offsets = [(0.27, 0.06, 0.73, 0.48), (0.27, 0.04, 0.73, 0.50)]

    edit_type_mapping = {}
    for i, d in enumerate(tqdm(data)):
        video_url, _type, w_offset, t_offset, prompt, prompt_edits, edit_types = d
        tmp_fname = tempfile.NamedTemporaryFile(suffix=".mp4").name
        response = requests.get(video_url, stream=True)
        assert response.status_code == 200, response.status_code
        with open(tmp_fname, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        reader = imageio.get_reader(tmp_fname)
        video = np.stack(list(iter(reader)))
        h, w = video.shape[1], video.shape[2]
        r_tl, c_tl, r_br, c_br = offsets[_type]
        video = video[:, int(h*r_tl):int(h*r_br), int(w*c_tl):int(w*c_br)]
        write_video(tmp_fname, video, fps=30)

        video = read_process_video(tmp_fname, fps=args.fps, resolution=args.resolution, offset_W=w_offset, offset_T=t_offset)

        for j, (prompt_edit, edit_type) in enumerate(zip(prompt_edits, edit_types)):
            base_fname = f"{i:02d}_{j:02d}" 
            edit_type_mapping[base_fname] = edit_type
            write_video(output_folder / f"{base_fname}.mp4", video, fps=4)

            with (output_folder / f"{base_fname}.txt").open("w") as f:
                f.write(f"{prompt}\n{prompt_edit}")
        os.system(f"rm {tmp_fname}")

    with (output_folder / 'edit_type_map.json').open('w') as f:
        json.dump(edit_type_mapping, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()
    main()
