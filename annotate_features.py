import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import sys


import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
#from lseg import LSegNet

import warnings
from pathlib import Path
from typing import Union

import torch
import torchvision
from typing_extensions import Literal
import time
import argparse
from DINO.collect_dino_features import *
from DINO.dino_wrapper import *

parser = argparse.ArgumentParser(description='Annotation script')
parser.add_argument('--cpu', action='store_true', default =False,  help='cpu mode')
parser.add_argument('--use_16bit', action='store_true',  default =False, help='16 bit dino mode')
parser.add_argument('--plot_similarity', action='store_true', default =False, help='16 bit dino mode')
parser.add_argument('--use_traced_model', action='store_true',  default =False, help='apply torch tracing')
parser.add_argument('--dino_strides', default=4, type=int , help='Strides for dino')
parser.add_argument('--desired_height', default=240, type=int, help='desired_height resulution')
parser.add_argument('--desired_width', default=320, type=int, help='desired_width resulution')
parser.add_argument('--queries_dir', default='./queries', help='The directory to collect the queries from')
parser.add_argument('--similarity_thresh', default=0.1, help='Threshold below which similarity scores are to be set to zero')
parser.add_argument('--path_to_images', default='./frames', help='The directory to collect the images from')
parser.add_argument('--path_to_video', default=None, type=str, help='Path to video file or camera index for frame extraction')
parser.add_argument('--text_query', default=None, type=str, help='Text query to use as class label for automated annotation')

args = parser.parse_args()

global clickx, clicky


def onclick(click):
    global clickx, clicky
    clickx = click.xdata
    clicky = click.ydata
    plt.close("all")


if __name__ == "__main__":
    # Process command-line args (if any)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = vars(args)

    print(f"[DEBUG] Using queries_dir: {cfg['queries_dir']}")
    print(f"[DEBUG] Using path_to_images: {cfg['path_to_images']}")
    print(f"[DEBUG] Using text_query: {cfg.get('text_query', None)}")



    # Process command-line args (if any)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = vars(args)

    print(f"[DEBUG] Using queries_dir: {cfg['queries_dir']}")
    print(f"[DEBUG] Using path_to_images: {cfg['path_to_images']}")
    print(f"[DEBUG] Using text_query: {cfg.get('text_query', None)}")

    if not os.path.exists(cfg['queries_dir']):
        os.mkdir(cfg['queries_dir'])

    # Automatically extract frames from video/camera if frames directory does not exist or is empty
    frames_dir = cfg['path_to_images']
    need_extract = False
    if not os.path.exists(frames_dir):
        print(f"[DEBUG] Frames directory {frames_dir} not found. Will extract frames...")
        os.makedirs(frames_dir, exist_ok=True)
        need_extract = True
    elif len(os.listdir(frames_dir)) == 0:
        print(f"[DEBUG] Frames directory {frames_dir} is empty. Will extract frames...")
        need_extract = True
    if need_extract:
        video_source = cfg.get('path_to_video', None)
        cap = None
        if video_source is not None:
            try:
                cam_index = int(video_source)
                print(f"[DEBUG] Attempting to open camera index: {cam_index}")
                cap = cv2.VideoCapture(cam_index)
            except ValueError:
                print(f"[DEBUG] Attempting to open video file: {video_source}")
                cap = cv2.VideoCapture(video_source)
        else:
            print("[DEBUG] Attempting to open default camera (index 0)")
            cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            print(f"[ERROR] Failed to open video source: {video_source}")
        else:
            frame_count = 0
            while frame_count < 10:
                ret, frame = cap.read()
                if not ret:
                    print(f"[ERROR] Failed to read frame {frame_count} from video source.")
                    break
                frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"[DEBUG] Saved frame {frame_count} to {frame_path}")
                frame_count += 1
            cap.release()
            print(f"[DEBUG] Extracted {frame_count} frames to {frames_dir}")

    # Always ensure frames directory exists
    if not os.path.exists(cfg['path_to_images']):
        os.makedirs(cfg['path_to_images'], exist_ok=True)

    model = get_dino_pixel_wise_features_model(cfg = cfg, device = device)
    annotated_feats = []
    annot_classes = []
    auto_mode = cfg.get('text_query') is not None

    with torch.no_grad():
        for imgname in os.listdir(cfg['path_to_images']):
            print(f"[DEBUG] Processing image: {imgname}")
            imgfile = os.path.join(cfg['path_to_images'], imgname)
            img = cv2.imread(imgfile)
            img = preprocess_frame(img, cfg=cfg)
            img_feat = model(img)
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
            h, w = img_feat_norm.shape[-2], img_feat_norm.shape[-1]
            if auto_mode:
                print(f"[DEBUG] Automated annotation mode. Annotated classes: {annot_classes}")
                clickx, clicky = w // 2, h // 2
                selected_embedding = img_feat_norm[0, :, clicky, clickx]
                annot_classes.append(cfg['text_query'])
                annotated_feats.append(selected_embedding.detach().cpu())
            else:
                # For image reference, save feature for each image
                selected_embedding = img_feat_norm[0, :, h // 2, w // 2]
                savefile = os.path.join(cfg['queries_dir'], f"{Path(imgname).stem}.pt")
                print(f"[DEBUG] Saving image reference feature: {savefile}")
                torch.save(selected_embedding.detach().cpu(), savefile)
        # Save all automated annotations
        if auto_mode:
            # Save each feature as a separate file in ./queries/<text_query_sanitized>/featN.pt
            key = cfg['text_query']
            query_folder = os.path.join(cfg['queries_dir'], key.replace(' ', '_'))
            if not os.path.exists(query_folder):
                os.makedirs(query_folder, exist_ok=True)
            if len(annotated_feats) > 0:
                for idx, feat in enumerate(annotated_feats):
                    savefile = os.path.join(query_folder, f"feat{idx}.pt")
                    print(f"[DEBUG] Saving annotation file: {savefile}")
                    torch.save(feat, savefile)
            else:
                print(f"[DEBUG] No features to save for query: {key}")