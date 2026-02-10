#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch

from model_args import segtracker_args, sam_args, aot_args
from SegTracker import SegTracker
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.check_call(cmd)


def crop_stereo_video_ffmpeg(in_path: str, out_path: str, view: str):
    """
    For side-by-side stereo videos: crop left half or right half.
    Requires ffmpeg available in PATH.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w <= 0 or h <= 0:
        raise RuntimeError("Failed to read video resolution for cropping.")

    half_w = w // 2
    if view == "left":
        x0 = 0
    elif view == "right":
        x0 = w - half_w
    else:
        raise ValueError(f"Unknown view: {view}")

    vf = f"crop={half_w}:{h}:{x0}:0"
    run_cmd([
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        out_path
    ])


def read_first_frame_rgb(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_fps(video_path: str, fallback: int = 8) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return fallback
    return int(round(float(fps)))


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def find_tracking_masks_dir(video_name: str) -> str:
    """
    tracking_objects_in_video() writes results into:
      ./tracking_results/<video_name>/<video_name>_masks/
    but we search robustly in case paths differ.
    """
    candidates = []
    # common location
    candidates += glob.glob(os.path.join("tracking_results", video_name, f"{video_name}_masks"))
    # fallback search
    candidates += glob.glob(os.path.join("tracking_results", "**", f"{video_name}_masks"), recursive=True)

    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        raise RuntimeError("Could not find output masks directory under ./tracking_results/. "
                           "tracking ran, but outputs not found.")
    # choose the most recently modified
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def save_masks_plain(masks_dir: str, out_dir: str):
    """
    Copies raw masks and also writes binary masks (nonzero => 255).
    """
    raw_out = os.path.join(out_dir, "masks_raw")
    bin_out = os.path.join(out_dir, "masks_bin")
    ensure_dir(raw_out)
    ensure_dir(bin_out)

    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*")))
    if not mask_files:
        raise RuntimeError(f"No mask files found in: {masks_dir}")

    for mf in mask_files:
        # keep original filename
        fname = os.path.basename(mf)

        # raw copy
        shutil.copy2(mf, os.path.join(raw_out, fname))

        # binary version
        im = Image.open(mf)
        # many masks are palette ('P'), keep ids as array
        arr = np.array(im)
        bin_arr = (arr != 0).astype(np.uint8) * 255
        Image.fromarray(bin_arr, mode="L").save(os.path.join(bin_out, fname))

    print(f"[ok] raw masks  -> {raw_out}")
    print(f"[ok] bin masks  -> {bin_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input mp4 path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--prompt", required=True, help="Text prompt for object (e.g., 'robot arm' or 'red cup')")

    # detection thresholds (same as UI defaults)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    # tracker args (reasonable defaults)
    parser.add_argument("--aot_model", default="r50_deaotl", choices=["deaotb", "deaotl", "r50_deaotl"])
    parser.add_argument("--sam_gap", type=int, default=9999, help="Large => don't keep adding new objects during tracking")
    parser.add_argument("--max_obj_num", type=int, default=2)
    parser.add_argument("--points_per_side", type=int, default=16)

    # AOT long-term memory
    parser.add_argument("--long_term_mem_gap", type=int, default=9999)
    parser.add_argument("--max_len_long_term", type=int, default=9999)

    # fps control (tracking_objects_in_video expects fps slider value)
    parser.add_argument("--fps", type=int, default=0, help="If 0, auto-read from video, fallback 8")

    # stereo cropping
    parser.add_argument("--stereo_view", default="full", choices=["full", "left", "right"],
                        help="If stereo side-by-side, choose which half to track")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    video_path = args.video
    video_name = Path(video_path).stem

    # Optional: crop stereo side-by-side
    if args.stereo_view in ("left", "right"):
        tmp_dir = os.path.join(args.out_dir, "_tmp")
        ensure_dir(tmp_dir)
        cropped_path = os.path.join(tmp_dir, f"{video_name}_{args.stereo_view}.mp4")
        crop_stereo_video_ffmpeg(video_path, cropped_path, args.stereo_view)
        video_path = cropped_path
        video_name = Path(video_path).stem

    # fps
    fps = args.fps if args.fps > 0 else get_fps(video_path, fallback=8)
    print(f"[info] fps = {fps}")

    # read first frame
    first_frame = read_first_frame_rgb(video_path)

    # configure args (mutate imported dicts)
    aot_args["model"] = args.aot_model
    aot_args["model_path"] = aot_model2ckpt[args.aot_model]
    aot_args["long_term_mem_gap"] = args.long_term_mem_gap
    aot_args["max_len_long_term"] = args.max_len_long_term

    segtracker_args["sam_gap"] = args.sam_gap
    segtracker_args["max_obj_num"] = args.max_obj_num
    sam_args["generator_args"]["points_per_side"] = args.points_per_side

    # init tracker
    print("[info] init SegTracker...")
    seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
    seg_tracker.restart_tracker()

    # detect + segment on first frame (text prompt)
    print(f"[info] detect_and_seg prompt: {args.prompt!r}")
    with torch.cuda.amp.autocast():
        pred_mask, annotated = seg_tracker.detect_and_seg(
            first_frame, args.prompt, args.box_threshold, args.text_threshold
        )

    if pred_mask is None or np.max(pred_mask) == 0:
        raise RuntimeError("Got empty mask from text prompt. Try lowering thresholds "
                           "or using a more specific prompt.")

    # save first mask for debugging
    Image.fromarray((pred_mask != 0).astype(np.uint8) * 255, mode="L").save(os.path.join(args.out_dir, "first_mask_bin.png"))
    Image.fromarray(annotated).save(os.path.join(args.out_dir, "first_frame_annotated.png"))
    print("[ok] wrote first_mask_bin.png and first_frame_annotated.png")

    # add reference and track
    print("[info] add reference + track...")
    frame_idx = 0
    with torch.cuda.amp.autocast():
        seg_tracker.restart_tracker()
        seg_tracker.add_reference(first_frame, pred_mask, frame_idx)
        seg_tracker.first_frame_mask = pred_mask

    # Run tracking (writes into ./tracking_results/...)
    out_video_file, out_mask_file = tracking_objects_in_video(seg_tracker, video_path, None, fps, 0)
    print(f"[info] tracking_objects_in_video returned:\n  video={out_video_file}\n  masks={out_mask_file}")

    # Find produced mask directory + export plain masks to out_dir
    masks_dir = find_tracking_masks_dir(video_name)
    print(f"[info] found masks dir: {masks_dir}")
    save_masks_plain(masks_dir, args.out_dir)

    print("[done]")


if __name__ == "__main__":
    main()
