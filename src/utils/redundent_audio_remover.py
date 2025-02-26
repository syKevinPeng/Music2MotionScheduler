# Generate Stick figure videos from the motorica dataset

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")
from smpl2motorica.utils import bvh
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer
import cv2
import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
import subprocess
import os 

if __name__ == "__main__":
    dataset_path = Path("/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} not found")
    mocap_path = dataset_path / "bvh"
    music_path = dataset_path / "wav"
    movement_fps = 120

    # get a list of bvh files that has corresponding music files
    bvh_files = list(mocap_path.glob("*.bvh"))
    print(f"total bvh files: {len(bvh_files)}")
    music_files = list(music_path.glob("*.wav"))
    bvh_files = [
        bvh for bvh in bvh_files if any(bvh.stem in music.stem for music in music_files)
    ]
    print(f"Found {len(bvh_files)} bvh files with corresponding music files")
    bvh_parser = BVHParser()
    result = []
    for bvh_file in bvh_files:
        print(f"Processing {bvh_file}")
        mocap_track = bvh_parser.parse(bvh_file)
        # get the length of the movement sequence
        length = len(mocap_track.values)
        # convert frames to seconds
        movement_length = length / movement_fps
        print(f"Total movement length: {movement_length} seconds")


        # get the length of the music sequence
        music_file = music_path / f"{bvh_file.stem}.wav"
        if not music_file.exists():
            raise Exception(f"Music file not found: {music_file}")
        audio = AudioFileClip(str(music_file))
        music_length = audio.duration
        print(f"Total music length: {music_length} seconds")
        result.append((bvh_file.stem, movement_length, music_length))

    # save as csv
    np.savetxt("movement_music_length.csv", result, delimiter=",", fmt="%s")

        