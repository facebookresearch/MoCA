# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse
from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
from PIL import Image
import imageio
import ffmpeg

OBJECT = "object"
STYLE = "style"
MOTION = "motion"
BACKGROUND = "background"
MULTI_SPATIAL = "multi_spatial"
MULTI_MOTION = "multi_motion"


# data follows the following format:
# [
#     (
#         <youtube_id>, [time offsets in original video (seconds) for each clip], [width offsets in original video (0-1) for each clip],
#         [source captions of each clip], [[edit captions for video clip 0], [... for clip 1], ...],
#         [[edit type for edit of video clip 0], [..., video clip 1], ...],
#     ) 
# ]
data = [
    ('WD8GsHUczI0', [301], [0.1], ["a white cat walks up to a black cat"], [["a white cat runs up to a tackles a black cat"]], [[MOTION]]),
    ( 
        '6mRgGoa08AQ', [67], [0.3], ["a bird sitting on a rock in a river"], 
        [
            ["a bird jumps from a rock into the river", "a bird takes flight off a rock", "a bird wearing a hat sitting on a rock in a river", 
             "a red cardinal sitting on a rock in a river", "a red cardinal takes flight off a rock"]
        ], 
        [
            [MOTION, MOTION, OBJECT, OBJECT, MULTI_MOTION]
        ],
    ),
    (
        '-aAcv7W9SJo', [50], [0.3], ["a dog standing in a grassy field"], 
        [["a dog is digging a hole in a grassy field", "a dog is running in a grassy field", "a cat standing in a grassy field", 
          "a dog standing in a grassy field on fire"]], 
        [[
            MULTI_MOTION, MOTION, OBJECT, BACKGROUND
        ]],
    ),
    (
        'w78UUl5E1Z8', [3, 26], [0.2, 0.2], ["a cat is standing in a cardboard box", "a cat is standing in front of a cardboard box"], 
        [["a cat jumps out of a cardboard box"], ["a cat jumps in a cardboard box"]], 
        [[MOTION], [MOTION]],
    ),
    (
        'NC_hmkNHI24', [22], [0.25], ["two monkeys sitting by the hot springs"], 
        [["two monkeys wrestling by hot springs", "two monkeys jump into the hot springs", "two monkeys sitting by a campfire"]], 
        [[MOTION, MOTION, BACKGROUND]],
    ),
    (
        'Ovmkw9Pmtzc', [6], [0.2], ["a baboon eating a fruit"], 
        [["a baboon drops fruit onto the ground", "a baboon waves its hands at the camera", "a baboon eating a green apple", "a baboon eating a banana"]], 
        [[MOTION, MOTION, STYLE, OBJECT]],
    ),
    (
        'IPfzM4l7bLo', [138], [0.2], ["a panda walking on tree branches"], 
        [["a panda sleeping on tree branches", "a panda slips and falls of the tree branches", "a grizzly bear walking on tree branches", "a panda falling from tree branches"]],
        [[MOTION, MOTION, OBJECT, MOTION]],
    ),
    (
        '-8zfnYsYFB0', [76, 107, 113], [0.2, 0.3, 0.3], 
        ["a lion laying in the grass", "a flamingo standing in the water", "a monkey walking around in grass"], 
        [
            ["a lion roaring on the grass", "a zebra laying in the grass", "a lion with a birthday hat dancing on the grass"], 
            ["a flamingo dunking its head in water to look for food", "a pink flamingo standing in the water", "a flamingo opening its wings while standing in the water.", "a pink flamingo opening its wings while standing in the water."], 
            ["a monkey picks up a banana in the grass", "a monkey ducks in the grass to hide itself", "a monkey jumping high in grass"]
        ], 
        [
            [MOTION, OBJECT, MULTI_MOTION],
            [MOTION, STYLE, MOTION, MULTI_MOTION],
            [MULTI_MOTION, MOTION, MOTION]
        ],
    ),
    (
        'fYCbiOEIcVM', [3], [0.25], ["an orangutan sitting in the river"], 
        [["an orangutan waves both its arms at the camera", "an orangutan does pushups", "an orangutan juggling fruits", 
          "an orangutan scratching its head", "an orangutan sitting next to the fire.", "an orangutan taking a bath in the river"]], 
        [[MOTION, MOTION, MULTI_MOTION, MOTION, BACKGROUND, MOTION]],
    ),
    (
        'Rab2eDwNxdY', [182], [0.3], ["a deer walking around green shrubbery"], 
        [["a deer dashes away", "a deer walks towards the camera", "a deer riding a skateboard around green shrubbery "]], 
        [[MOTION, MOTION, MULTI_MOTION]],
    ),
    (
        'FPU3MkaT_9k', [86], [0.15], ["a panda sitting down and eating from a pile of bamboo"], 
        [["a panda falls over onto a pile of bamboo", "a panda throws bamboo leaves out of its hand", 
          "a panda with a cowboy hat sitting down and eating from a pile of bamboo", "a panda sitting down and eating from a pile of dried grass", 
          "a panda sitting down and playing in a pile of colorful ribbons"]], 
        [[MOTION, MOTION, OBJECT, STYLE, STYLE]],
    ),
    (
        'nA1jBUEzgSQ', [37], [0.2], ["a duck swimming on water"], 
        [["a duck dunks its head underwater", "a swan swimming on water", "a duck sitting on a piece of wood floating on water", 
          "a duck shaking wings on water", "a duck swimming on water with big waves"]], 
        [[MOTION, OBJECT, OBJECT, MOTION, BACKGROUND]],
    ),
    (
        'SQ9LIt2dpQQ', [10], [0.2], ["a duck floating on a river"], 
        [["a duck flies away from the water", "a duck dives underwater", "a camera zooms in on a duck floating on a river"]], 
        [[MOTION, MOTION, MOTION]],
    ),
    (
        'WmmGvnuAF18', [48], [0.2], ["a duckling looking for food in grass"], 
        [["zooming out from a duckling looking for food in grass", "a duckling dashes away", 
          "a duckling jumping on a pile of leaves", "a duckling jumping on grass"]], 
        [[MOTION, MOTION, MULTI_MOTION, MOTION]],
    ),
    (
        'yrAc0EIiHB0', [148], [0.2], ["a squirrel in the grass"], 
        [["a squirrel burying nuts in the grass", "a startled squirrel jumps away"]], 
        [[MULTI_MOTION, MOTION]],
    ),
    (
        'dT4wnmFXcGY', [4], [0.4], ["a squirrel stands up to reach branch"], 
        [["a squirrel scurries up a nearby tree"]], 
        [[MOTION]],
    ),
    (
        'pLAHIjC8MIs', [52], [0.2], ["a boar sniffs the dirt looking for food"], 
        [["a boar digs a hole in the ground", "a boar rolls over on its belly", "a boar jumps up and down"]], 
        [[MULTI_MOTION, MOTION, MOTION]],
    ),
    (
        'nc2R3JIBpd4', [15], [0.25], ["a dog is playing near a beach"], 
        [["a dog shakes water off itself to dry off", "a dog drinks nearby water", "a dog is playing with a ball near a beach", "a dog is catching a frisbee near a beach"]], 
        [[MOTION, MOTION, MULTI_MOTION, MULTI_MOTION]],
    ),
    (
        'yC5TCzQ5V3g', [54], [0.2], ["a dog rolling around on the grass"], 
        [["a dog standing on the grass", "a dog running on the grass"]], 
        [[MOTION, MOTION]],
    ),
    (
        'O7HE0dvNx8c', [7], [0.3], ["a scene of a calm lake"], 
        [["a scene of a water geyser erupting from a lake", "a scene of a left to right pan of a calm lake", 
          "a scene of a right to left pan of a calm lake", "a time-lapse of a calm lake", "a scene of a lake with crashing waves"]], 
        [[MOTION, MOTION, MOTION, MOTION, STYLE]],
    ),
    (
        'UmtZLpHPRCs', [72], [0.2], ["bright red leaves on a tree during autumn"], 
        [["bright red leaves on a tree during autumn, windy day, rustling leaves", 
          "bright red leaves fall off a tree during autumn", "green leaves on a tree during summer"]], 
        [[MOTION, MOTION, STYLE]],
    ),
    (
        'rreEpKo3o_Y', [33], [0.2], ["a bright pink flower"], 
        [["timelapse of a bright pink flower fully blooming", "timelapse of an orange lily fully blooming"]], 
        [[MOTION, MULTI_MOTION]],
    ),
    (
        'zYBq6V-3BpM', [78, 95], [0.2, 0.3], ["fresh apricots hanging off a tree", "fresh apricots hanging off a tree"], 
        [
            ["ripe apricots fall off a tree", "ripe apples fall off a tree"], 
            ["ripe apricots fall off a tree", "fresh apricots hanging off a tree, windy day, rustling", 
             "fresh apples hanging off a tree, windy day, rustling", "fresh apples hanging off a tree"]
        ], 
        [
            [MOTION, OBJECT],
            [MOTION, MOTION, MULTI_MOTION, OBJECT]
        ],
    ),
    (
        'AMd0FIM0Lew', [123], [0.2], ["a person standing in a bear costume"], 
        [["a person doing jumping jacks in a bear costume", "a person running in a bear costume", "a person doing pushups in a bear costume"]], 
        [[MOTION, MOTION, MOTION]],
    ),
    (
        'apKOwzzZY38', [95], [0.1], ["fish swimming around in a man-made pond"], 
        [["fish jump out of a man-made pond"]], 
        [[MOTION]],
    ),
    (
        'q4yHz3ysMUk', [101], [0.1], ["goldfish swimming around in a lake"], 
        [["goldfish rush towards breadcrumbs thrown on the lake", "blue fish swimming around in a lake"]], 
        [[MULTI_MOTION, STYLE]],
    ),
    (
        'T0DV3BriqZs', [3], [0.0], ["a turtle laying on a green floor"], 
        [["a turtle laying on a green floor retracts into its shell", "a turtle laying on a green floor rushes towards some apple slices on the ground"]], 
        [[MOTION, MULTI_MOTION]],
    ),
    (
        'YSXzPACM6gs', [270, 335], [0.2, 0.0], ["a calm view of the ocean and a nearby island", "an ocean view while standing on the side of a boat"], 
        [["a stormy view of the ocean and nearby island, waves crashing"], ["an ocean view while standing on the side of a boat, waves crash onto the boat"]], 
        [[STYLE], [MOTION]],
    ),
    (
        'QHD7sDM32Sg', [72], [0.2], ["looking down into calm waters below from a nearby cliff"], 
        [["jumping down into calm water below from a nearby cliff"]], 
        [[MOTION]],
    ),
    (
        'wNITe1mUNxw', [36], [0.2], ["a white jeep driving down a gravel road"], 
        [["a white jeep driving down an extremely bumpy gravel road", "a white jeep driving down a dirt road", 
          "a white jeep comes to a stop while driving down a gravel road", "a white jeep driving down a gravel road while all trees burn in fire"]], 
        [[STYLE, OBJECT, MOTION, BACKGROUND]],
    ),
    (
        'r7gL7fORf24', [45], [0.2], ["riding a boat over the ocean"], 
        [["huge waves crash while riding a boat over the ocean"]], 
        [[MOTION]],
    ),
    (
        'ce-KW87rKGM', [2], [0.3], ["dashcam view of a person driving down a highway"], 
        [["dashcam view of a person coming to a stop on a highway", "dashcam view of a person veering sidways while driving down a highway", 
          "dashcam view of a person driving down a highway while raining"]], 
        [[MOTION, MOTION, BACKGROUND]],
    ),
    (
        'zehsnFf1Ylo', [38], [0.0], ["a walk around view of a parked motorcycle"], 
        [["a walk around view of a parked motorcycle as it tips over"]], 
        [[MOTION]],
    ),
    (
        'nr8AeHTJ7mA', [5], [0.3], ["a brown rabbit resting in its cage"], 
        [["a brown rabbit eating a carrot in its cage", "a brown rabbit hopping around in its cage", "a white rabbit resting in its cage"]], 
        [[OBJECT, MOTION, STYLE]],
    ),
    (
        'GtTEPLCKENg', [22], [0.15], ["a silver car coming to a stop"], 
        [["a silver car zooms down the road", "a red porsche comining to a stop"]], 
        [[MOTION, OBJECT]],
    ),
    (
        'kULvMcStIUY', [44], [0.2], ["sideview of a racecar driving down the race track"], 
        [["sideview of a racecar drifting off onto the grass near the race track", "sideview of a racecar driving down the race track during a snowstorm"]], 
        [[MOTION, BACKGROUND]],
    ),
    (
        'DykXOiH9Kos', [127], [0.1], ["a speeding car slows down to turn a narrow bend"], 
        [["a speeding car tips over as it tries to turn a narrow bend", "a speeding car made of lego slows down to turn a narrow bend", 
          "a speeding car with wings flies to the sky"]], 
        [[MOTION, STYLE, MULTI_MOTION]],
    ),
    
]

def read_video(input_file, fps=4, resolution=256, T=2, offset_W=0, offset_T=0):
    input_file = str(input_file)
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

def main():
    data_dir = Path(args.output_folder)
    data_dir.mkdir(parents=True, exist_ok=True)

    edit_type_mapping = {}
    for i, (youtube_id, clip_starts, clip_offsets, captions, edit_captions_all, edit_types_all) in enumerate(tqdm(data)):
        save_fname = data_dir / f'{youtube_id}.mp4'
        if not save_fname.exists():
            cmd = f"yt-dlp -f 'best[ext=mp4]' -o {save_fname} https://www.youtube.com/watch?v={youtube_id}"
            print(cmd)
            os.system(cmd)
        for j, (clip_start, clip_offset, caption, edit_captions, edit_types) in enumerate(zip(clip_starts, clip_offsets, captions, edit_captions_all, edit_types_all)):
            video = read_video(save_fname, resolution=args.resolution, fps=args.fps, offset_W=clip_offset, offset_T=clip_start)
            for k, (edit_caption, edit_type) in enumerate(zip(edit_captions, edit_types)):
                edit_type_mapping[f'{youtube_id}_{j:02d}_{k:02}'] = edit_type
                imageio.mimsave(data_dir / f'{youtube_id}_{j:02d}_{k:02}.mp4', video, fps=args.fps)
                with (data_dir / f'{youtube_id}_{j:02}_{k:02}.txt').open('w') as f:
                    f.write(f"{caption}\n{edit_caption}")
    with (data_dir / 'edit_type_map.json').open('w') as f:
        json.dump(edit_type_mapping, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()
    main()
