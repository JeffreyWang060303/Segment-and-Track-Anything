#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from STAGE_1_samt_masking import run_samt_mask_pipeline

res = run_samt_mask_pipeline(
    repo_root="/viscam/u/jeffreywang0303/projects/Inpainting-Objects/Segment-and-Track-Anything",
    video_path="/viscam/data/inpainting_dataset_jw/droid/2026-02-06/416x240x81/sample_00000/wrist_view/stereo.mp4",
    out_dir="/viscam/u/jeffreywang0303/projects/Inpainting-Objects/experiments/E6",

    view="wrist",            # ext1/ext2/wrist
    input_type="stereo",    # original/stereo

    mask_scale=1.01,

    relevant_objects=["white mug"],
    always_relevant_objects=["purple mat"],
    irrelevant_objects=["white tube"],

    # 可选：不填就用默认（ext:0.25, wrist:0.3）
    # box_threshold=0.25,
    text_threshold=0.25,

    # wrist 扫描策略（只在 view="wrist" 生效）
    wrist_min_conf=0.3,
    wrist_scan_stride=1,

    sam_gap=9999,
    max_obj_num=1,
    points_per_side=16,
)

print("DONE:", res["mask_video"])
