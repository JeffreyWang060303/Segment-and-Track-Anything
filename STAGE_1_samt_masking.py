#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import subprocess
import json
from pathlib import Path
from contextlib import contextmanager
from collections import Counter

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

from model_args import segtracker_args, sam_args, aot_args
from SegTracker import SegTracker
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video


# ----------------------------
# Configuration Parameters
# ----------------------------

class MaskValidationConfig:
    """Parameters for mask validation"""
    # Multi-frame selection (wrist mode only)
    TOP_N_CANDIDATES = 100  # Consider top N confident frames
    MIN_CLUSTER_SIZE = 3   # Minimum frames that should agree on mask location
    
    # Stereo consistency check (all modes)
    POSITIONAL_SIMILARITY_THRESHOLD = 0.50  # 80% of frames must match positionally
    MAX_FAILED_FRAMES = 50  # Max frames allowed to fail positional check
    
    # Confidence thresholds
    MIN_CONFIDENCE_WRIST = 0.3
    MIN_CONFIDENCE_EXT = 0.25
    
    # Blanking
    BLANK_COLOR = [128, 128, 128]  # Gray color for blanked regions


# ----------------------------
# utils
# ----------------------------

@contextmanager
def pushd(new_dir: str):
    old = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.check_call(cmd)


def read_frame_rgb(video_path: str, frame_idx: int):
    """Read specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_first_frame_rgb(video_path: str):
    return read_frame_rgb(video_path, 0)


def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def get_fps(video_path: str, fallback: int = 8) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return fallback
    return int(round(float(fps)))


def get_video_hw(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError("Failed to read video resolution.")
    return h, w


def crop_stereo_video_ffmpeg(in_path: str, out_path: str, view: str):
    """Side-by-side stereo crop using ffmpeg."""
    h, w = get_video_hw(in_path)
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


def scale_binary_mask(mask01: np.ndarray, scale: float) -> np.ndarray:
    """
    mask01: uint8 {0,1}, shape (H,W)
    scale: e.g. 1.01 enlarge, 0.98 shrink
    Scales around the mask bbox center and returns same-size binary mask.
    """
    if scale is None:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return mask01

    H, W = mask01.shape
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return mask01

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    crop = mask01[y0:y1+1, x0:x1+1]
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))

    resized = cv2.resize(crop.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized = (resized > 0).astype(np.uint8)

    # place back centered at (cx, cy)
    out = np.zeros((H, W), dtype=np.uint8)

    nx0 = int(round(cx - new_w / 2.0))
    ny0 = int(round(cy - new_h / 2.0))
    nx1 = nx0 + new_w
    ny1 = ny0 + new_h

    sx0 = max(0, nx0)
    sy0 = max(0, ny0)
    sx1 = min(W, nx1)
    sy1 = min(H, ny1)

    rx0 = sx0 - nx0
    ry0 = sy0 - ny0
    rx1 = rx0 + (sx1 - sx0)
    ry1 = ry0 + (sy1 - sy0)

    if sx1 > sx0 and sy1 > sy0:
        out[sy0:sy1, sx0:sx1] = resized[ry0:ry1, rx0:rx1]

    return out


def compute_mask_center(mask01: np.ndarray):
    """Compute centroid of binary mask."""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return (np.mean(xs), np.mean(ys))


def compute_mask_bbox(mask01: np.ndarray):
    """Compute bounding box of mask. Returns (x1, y1, x2, y2) or None."""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return (xs.min(), ys.min(), xs.max(), ys.max())


def compute_bbox_iou(bbox1, bbox2):
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_mask_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_positional_similarity(mask1, mask2):
    """
    Compute positional similarity based on RELATIVE positions in each frame.
    For stereo videos, compares the relative position (normalized coordinates)
    rather than absolute pixel overlap.
    
    Returns similarity score 0-1.
    """
    bbox1 = compute_mask_bbox(mask1)
    bbox2 = compute_mask_bbox(mask2)
    
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    H1, W1 = mask1.shape
    H2, W2 = mask2.shape
    
    # Compute normalized center positions (0-1 range)
    c1 = compute_mask_center(mask1)
    c2 = compute_mask_center(mask2)
    
    if c1 is None or c2 is None:
        return 0.0
    
    # Normalize to [0, 1] range
    norm_cx1 = c1[0] / W1
    norm_cy1 = c1[1] / H1
    
    norm_cx2 = c2[0] / W2
    norm_cy2 = c2[1] / H2
    
    # Compute normalized bbox sizes
    norm_w1 = (bbox1[2] - bbox1[0]) / W1
    norm_h1 = (bbox1[3] - bbox1[1]) / H1
    
    norm_w2 = (bbox2[2] - bbox2[0]) / W2
    norm_h2 = (bbox2[3] - bbox2[1]) / H2
    
    # Center distance in normalized space
    center_dist = np.sqrt((norm_cx1 - norm_cx2)**2 + (norm_cy1 - norm_cy2)**2)
    # Diagonal of unit square is sqrt(2)
    center_similarity = 1.0 - min(center_dist / np.sqrt(2), 1.0)
    
    # Size similarity (ratio of sizes, symmetric)
    size_ratio_w = min(norm_w1, norm_w2) / (max(norm_w1, norm_w2) + 1e-6)
    size_ratio_h = min(norm_h1, norm_h2) / (max(norm_h1, norm_h2) + 1e-6)
    size_similarity = (size_ratio_w + size_ratio_h) / 2.0
    
    # Combined similarity: 70% position + 30% size
    similarity = 0.7 * center_similarity + 0.3 * size_similarity
    
    return similarity


def make_unique_video_symlink(src_video: str, tmp_dir: str, stem: str) -> str:
    """
    tracking_objects_in_video writes to ./tracking_results/<video_stem>/...
    We ensure each prompt run has a unique stem by symlinking/copying.
    """
    ensure_dir(tmp_dir)
    ext = Path(src_video).suffix
    dst = os.path.join(tmp_dir, f"{stem}{ext}")
    # Replace existing
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.symlink(src_video, dst)
    except OSError:
        # fallback to copy if symlink not allowed
        shutil.copy2(src_video, dst)
    return dst


def find_tracking_masks_dir(repo_root: str, video_stem: str) -> str:
    """
    Locate tracking output mask folder under repo_root/tracking_results.
    """
    base = os.path.join(repo_root, "tracking_results")
    candidates = []
    candidates += glob.glob(os.path.join(base, video_stem, f"{video_stem}_masks"))
    candidates += glob.glob(os.path.join(base, "**", f"{video_stem}_masks"), recursive=True)
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        raise RuntimeError(f"Could not find output masks directory for stem={video_stem} under {base}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_binary_masks_from_dir(masks_dir: str) -> list:
    """
    Returns list of uint8 {0,1} masks sorted by filename.
    """
    files = sorted(glob.glob(os.path.join(masks_dir, "*")))
    if not files:
        raise RuntimeError(f"No mask files in: {masks_dir}")
    out = []
    for f in files:
        arr = np.array(Image.open(f))
        out.append((arr != 0).astype(np.uint8))
    return out


def write_mask_video(mask_frames01: list, out_mp4: str, fps: int):
    """
    mask_frames01: list of (H,W) uint8 {0,1}; writes mp4 grayscale (0/255)
    """
    ensure_dir(str(Path(out_mp4).parent))
    H, W = mask_frames01[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, float(fps), (W, H), True)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    for m01 in mask_frames01:
        m255 = (m01.astype(np.uint8) * 255)
        bgr = cv2.cvtColor(m255, cv2.COLOR_GRAY2BGR)
        vw.write(bgr)
    vw.release()


def write_mask_pngs(mask_frames01: list, out_dir: str):
    ensure_dir(out_dir)
    for i, m01 in enumerate(mask_frames01):
        m255 = (m01.astype(np.uint8) * 255)
        Image.fromarray(m255, mode="L").save(os.path.join(out_dir, f"{i:05d}.png"))


# ----------------------------
# Multi-frame best mask selection using KNN-like voting
# ----------------------------

def find_best_frame_mask_multiframe_knn(
    repo_root: str,
    video_path: str,
    prompt: str,
    out_dir: str,
    *,
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    min_confidence: float = 0.25,
    sample_stride: int = 1,
    exclusion_mask: np.ndarray = None,
    top_n: int = None,
    min_cluster_size: int = None,
) -> tuple:
    """
    Scan frames using Grounding DINO + SAM with KNN-like voting.
    Returns the mask that appears most frequently in top-N confident frames.
    """
    ensure_dir(out_dir)
    
    if top_n is None:
        top_n = MaskValidationConfig.TOP_N_CANDIDATES
    if min_cluster_size is None:
        min_cluster_size = MaskValidationConfig.MIN_CLUSTER_SIZE
    
    # Load Grounding DINO model
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[multi-frame-knn] Loading Grounding DINO + SAM models")
    processor = AutoProcessor.from_pretrained(model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # Load SAM
    sam_checkpoint = None
    sam_model_type = None
    
    possible_sams = [
        ("vit_h", "sam_vit_h_4b8939.pth"),
        ("vit_l", "sam_vit_l_0b3195.pth"),
        ("vit_b", "sam_vit_b_01ec64.pth"),
    ]
    
    for model_type, filename in possible_sams:
        paths_to_try = [
            os.path.join(repo_root, "ckpt", filename),
            os.path.join(repo_root, filename),
            f"./ckpt/{filename}",
            f"./{filename}",
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                sam_checkpoint = path
                sam_model_type = model_type
                break
        
        if sam_checkpoint:
            break
    
    if not sam_checkpoint:
        raise FileNotFoundError(
            "Could not find SAM checkpoint. Please ensure one of these exists:\n" +
            "\n".join(f"  - {repo_root}/ckpt/{f}" for _, f in possible_sams)
        )
    
    print(f"[multi-frame-knn] Loading SAM ({sam_model_type}) from: {sam_checkpoint}")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    
    total_frames = get_video_frame_count(video_path)
    if total_frames > 300:
        frame_indices = list(range(0, total_frames, max(1, total_frames // 100)))
    else:
        frame_indices = list(range(0, total_frames, sample_stride))
    
    print(f"[multi-frame-knn] Scanning {len(frame_indices)}/{total_frames} frames for: '{prompt}'")
    if exclusion_mask is not None:
        print(f"[multi-frame-knn] Blanking out exclusion regions before detection")
    
    candidates = []  # (frame_idx, box, confidence, mask01, center, bbox)
    
    for idx, frame_idx in enumerate(frame_indices):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(frame_indices)} frames...")
        
        try:
            frame_rgb = read_frame_rgb(video_path, frame_idx)
            H, W = frame_rgb.shape[:2]
            
            # Blank out exclusion regions
            if exclusion_mask is not None:
                if exclusion_mask.shape != (H, W):
                    exclusion_resized = cv2.resize(
                        exclusion_mask.astype(np.uint8), 
                        (W, H), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    exclusion_resized = exclusion_mask
                
                frame_rgb_masked = frame_rgb.copy()
                frame_rgb_masked[exclusion_resized > 0] = MaskValidationConfig.BLANK_COLOR
            else:
                frame_rgb_masked = frame_rgb
            
            pil_img = Image.fromarray(frame_rgb_masked)
            
            # Run Grounding DINO
            inputs = processor(images=pil_img, text=[[prompt]], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = gd_model(**inputs)
            
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[pil_img.size[::-1]]
            )
            
            result = results[0]
            
            if len(result["scores"]) == 0:
                continue
            
            # Get best detection
            best_idx = result["scores"].argmax()
            confidence = result["scores"][best_idx].item()
            box = result["boxes"][best_idx].tolist()
            
            if confidence < min_confidence:
                continue
            
            # Use SAM on original image
            sam_predictor.set_image(frame_rgb)
            box_xyxy = np.array(box)
            
            masks_sam, scores_sam, _ = sam_predictor.predict(
                box=box_xyxy,
                multimask_output=False
            )
            
            mask01 = masks_sam[0].astype(np.uint8)
            center = compute_mask_center(mask01)
            bbox = compute_mask_bbox(mask01)
            
            if center is not None and bbox is not None:
                candidates.append((frame_idx, box, confidence, mask01, center, bbox))
                print(f"  ✓ Frame {frame_idx}: conf={confidence:.3f} at box={[int(x) for x in box]}")
                    
        except Exception as e:
            if idx % 50 == 0:
                print(f"  ✗ Frame {frame_idx}: {str(e)[:60]}")
            continue
    
    # Cleanup models
    del gd_model
    del processor
    del sam
    del sam_predictor
    torch.cuda.empty_cache()
    
    print(f"[multi-frame-knn] Found {len(candidates)} valid detections")
    
    if not candidates:
        print(f"[multi-frame-knn] No valid detection found for '{prompt}'")
        return None, None, None
    
    # KNN-like voting: Take top N by confidence, cluster by position
    candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)
    top_candidates = candidates_sorted[:min(top_n, len(candidates_sorted))]
    
    print(f"[multi-frame-knn] Clustering top {len(top_candidates)} candidates")
    
    # Cluster candidates by positional similarity
    clusters = []
    
    for i, cand_i in enumerate(top_candidates):
        _, _, _, mask_i, _, _ = cand_i
        
        # Try to add to existing cluster
        added = False
        for cluster in clusters:
            # Check if similar to any member of this cluster
            for j in cluster:
                _, _, _, mask_j, _, _ = top_candidates[j]
                similarity = compute_positional_similarity(mask_i, mask_j)
                
                if similarity > 0.5:
                    cluster.append(i)
                    added = True
                    break
            if added:
                break
        
        if not added:
            clusters.append([i])
    
    # Find largest cluster
    largest_cluster = max(clusters, key=len)
    
    print(f"[multi-frame-knn] Found {len(clusters)} clusters, largest has {len(largest_cluster)} members")
    
    if len(largest_cluster) < min_cluster_size:
        print(f"[multi-frame-knn] WARNING: Largest cluster size {len(largest_cluster)} < min {min_cluster_size}")
        print(f"[multi-frame-knn] Proceeding anyway with best available")
    
    # Select the highest confidence member from the largest cluster
    cluster_members = [top_candidates[i] for i in largest_cluster]
    best = max(cluster_members, key=lambda x: x[2])
    
    frame_idx, box, confidence, mask01, center, bbox = best
    
    print(f"[multi-frame-knn] Selected frame {frame_idx}:")
    print(f"  - Confidence: {confidence:.3f}")
    print(f"  - Cluster size: {len(largest_cluster)}/{len(top_candidates)}")
    
    # Save debug visualization
    try:
        frame_rgb = read_frame_rgb(video_path, frame_idx)
        overlay = frame_rgb.copy()
        overlay[mask01 > 0] = (overlay[mask01 > 0] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
        
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        Image.fromarray(overlay).save(
            os.path.join(out_dir, f"best_frame_{frame_idx:04d}_conf{confidence:.2f}_{safe_stem(prompt)}.png")
        )
        Image.fromarray((mask01 * 255).astype(np.uint8), mode='L').save(
            os.path.join(out_dir, f"best_mask_{frame_idx:04d}_{safe_stem(prompt)}.png")
        )
        
        if exclusion_mask is not None:
            frame_rgb_debug = read_frame_rgb(video_path, frame_idx)
            if exclusion_mask.shape != frame_rgb_debug.shape[:2]:
                exclusion_resized = cv2.resize(
                    exclusion_mask.astype(np.uint8), 
                    (frame_rgb_debug.shape[1], frame_rgb_debug.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                exclusion_resized = exclusion_mask
            frame_rgb_debug[exclusion_resized > 0] = MaskValidationConfig.BLANK_COLOR
            Image.fromarray(frame_rgb_debug).save(
                os.path.join(out_dir, f"masked_input_{frame_idx:04d}_{safe_stem(prompt)}.png")
            )
    except Exception as e:
        print(f"Warning: Could not save debug images: {e}")
    
    return frame_idx, mask01, confidence


# ----------------------------
# Stereo consistency validation
# ----------------------------

def validate_stereo_consistency(
    left_masks: list,
    right_masks: list,
    object_name: str,
) -> tuple:
    """
    Check if left and right masks are positionally consistent across frames.
    Returns: (passed: bool, metrics: dict)
    """
    if len(left_masks) != len(right_masks):
        print(f"[stereo-validation] {object_name}: Frame count mismatch")
        return False, {"error": "frame_count_mismatch"}
    
    total_frames = len(left_masks)
    passed_frames = 0
    failed_frames = 0
    similarities = []
    
    for t in range(total_frames):
        similarity = compute_positional_similarity(left_masks[t], right_masks[t])
        similarities.append(similarity)
        if similarity >= MaskValidationConfig.POSITIONAL_SIMILARITY_THRESHOLD:
            passed_frames += 1
        else:
            failed_frames += 1
    
    pass_rate = passed_frames / total_frames if total_frames > 0 else 0
    avg_similarity = np.mean(similarities) if similarities else 0
    
    print(f"[stereo-validation] {object_name}:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Passed: {passed_frames} ({pass_rate:.1%})")
    print(f"  - Failed: {failed_frames}")
    print(f"  - Avg similarity: {avg_similarity:.3f}")
    
    metrics = {
        "total_frames": total_frames,
        "passed_frames": passed_frames,
        "failed_frames": failed_frames,
        "pass_rate": pass_rate,
        "avg_similarity": avg_similarity,
    }
    
    # Check if failed frames are within tolerance
    if failed_frames <= MaskValidationConfig.MAX_FAILED_FRAMES:
        print(f"  - ✓ PASS (failed frames {failed_frames} <= {MaskValidationConfig.MAX_FAILED_FRAMES})")
        return True, metrics
    else:
        print(f"  - ✗ FAIL (failed frames {failed_frames} > {MaskValidationConfig.MAX_FAILED_FRAMES})")
        return False, metrics


# ----------------------------
# core: one prompt -> tracked masks
# ----------------------------

def track_one_prompt_to_masks01(
    repo_root: str,
    video_path: str,
    prompt: str,
    out_dir: str,
    *,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    aot_model: str = "r50_deaotl",
    sam_gap: int = 9999,
    max_obj_num: int = 2,
    points_per_side: int = 16,
    long_term_mem_gap: int = 9999,
    max_len_long_term: int = 9999,
    fps: int = 0,
    mask_scale: float = 1.0,
    debug_prefix: str = "",
    use_multiframe_selection: bool = False,
    exclusion_masks: list = None,
) -> list:
    """
    Runs SAMT for a single prompt and returns per-frame binary masks (uint8 {0,1}).
    """
    ensure_dir(out_dir)
    if fps <= 0:
        fps = get_fps(video_path, fallback=8)

    tmp_dir = os.path.join(out_dir, "_tmp_inputs")
    unique_stem = f"{Path(video_path).stem}__{debug_prefix}{safe_stem(prompt)}"
    unique_video = make_unique_video_symlink(video_path, tmp_dir, unique_stem)
    video_stem = Path(unique_video).stem

    with pushd(repo_root):
        if use_multiframe_selection:
            print(f"[wrist mode] Multi-frame KNN selection for: {prompt}")
            
            # Use first frame exclusion for detection
            exclusion_mask_first = exclusion_masks[0] if exclusion_masks and len(exclusion_masks) > 0 else None
            
            result = find_best_frame_mask_multiframe_knn(
                repo_root=repo_root,
                video_path=unique_video,
                prompt=prompt,
                out_dir=out_dir,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                min_confidence=box_threshold,
                exclusion_mask=exclusion_mask_first,
            )
            
            if result[0] is None:
                print(f"[wrist mode] No valid mask found for '{prompt}', returning empty masks")
                total_frames = get_video_frame_count(unique_video)
                h, w = get_video_hw(unique_video)
                return [np.zeros((h, w), dtype=np.uint8) for _ in range(total_frames)]
            
            best_frame_idx, pred_mask_01, confidence = result
            reference_frame = read_frame_rgb(unique_video, best_frame_idx)
            pred_mask_scaled = scale_binary_mask(pred_mask_01, mask_scale)
            
        else:
            # Original: use first frame with SegTracker
            first_frame = read_first_frame_rgb(unique_video)
            reference_frame = first_frame
            best_frame_idx = 0
            
            seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
            seg_tracker.restart_tracker()

            print(f"[prompt] {prompt}")
            with torch.cuda.amp.autocast():
                pred_mask, annotated = seg_tracker.detect_and_seg(
                    first_frame, prompt, box_threshold, text_threshold
                )

            if pred_mask is None or np.max(pred_mask) == 0:
                raise RuntimeError(f"Empty mask for prompt={prompt!r}")

            m01 = (pred_mask != 0).astype(np.uint8)
            pred_mask_scaled = scale_binary_mask(m01, mask_scale)
            
            Image.fromarray(pred_mask_scaled * 255, mode="L").save(
                os.path.join(out_dir, f"{debug_prefix}first_mask_bin__{safe_stem(prompt)}.png")
            )

        # Configure tracking
        aot_args["model"] = aot_model
        aot_args["model_path"] = aot_model2ckpt[aot_model]
        aot_args["long_term_mem_gap"] = long_term_mem_gap
        aot_args["max_len_long_term"] = max_len_long_term

        segtracker_args["sam_gap"] = sam_gap
        segtracker_args["max_obj_num"] = max_obj_num
        sam_args["generator_args"]["points_per_side"] = points_per_side

        seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
        seg_tracker.restart_tracker()

        with torch.cuda.amp.autocast():
            seg_tracker.add_reference(reference_frame, pred_mask_scaled, best_frame_idx)
            seg_tracker.first_frame_mask = pred_mask_scaled

        # Always track from frame 0
        tracking_objects_in_video(seg_tracker, unique_video, None, fps, 0)

        masks_dir = find_tracking_masks_dir(repo_root, video_stem)
        masks01 = load_binary_masks_from_dir(masks_dir)

    return masks01


def _process_one_video_multiobj(
    *,
    repo_root: str,
    video_path: str,
    out_dir: str,
    relevant_objects: list,
    irrelevant_objects: list,
    always_relevant_objects: list,
    mask_scale: float,
    box_threshold: float,
    text_threshold: float,
    aot_model: str,
    sam_gap: int,
    max_obj_num: int,
    points_per_side: int,
    long_term_mem_gap: int,
    max_len_long_term: int,
    fps: int,
    use_multiframe_selection: bool = False,
) -> tuple:
    """
    Process video with multiple object prompts.
    Returns: (final_masks, object_masks_dict, successfully_masked_list)
    """
    ensure_dir(out_dir)

    keep_union = None
    remove_union = None
    
    # Get video dimensions
    h, w = get_video_hw(video_path)
    total_frames = get_video_frame_count(video_path)
    
    # Dictionary to store individual object masks
    object_masks_dict = {}  # {object_name: masks01}
    
    def run_and_union_relevant(prompts, union_target_name: str):
        nonlocal keep_union
        
        for i, p in enumerate(prompts):
            p_out = os.path.join(out_dir, f"prompt_{union_target_name}_{i:02d}__{safe_stem(p)}")
            ensure_dir(p_out)

            masks01 = track_one_prompt_to_masks01(
                repo_root=repo_root,
                video_path=video_path,
                prompt=p,
                out_dir=p_out,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                aot_model=aot_model,
                sam_gap=sam_gap,
                max_obj_num=max_obj_num,
                points_per_side=points_per_side,
                long_term_mem_gap=long_term_mem_gap,
                max_len_long_term=max_len_long_term,
                fps=fps,
                mask_scale=mask_scale,
                debug_prefix=f"{union_target_name}_",
                use_multiframe_selection=use_multiframe_selection,
                exclusion_masks=None,
            )

            # Store individual object masks
            object_masks_dict[p] = masks01
            
            # Union
            if keep_union is None:
                keep_union = [m.copy() for m in masks01]
            else:
                for t in range(min(len(keep_union), len(masks01))):
                    keep_union[t] = np.maximum(keep_union[t], masks01[t])
            
            print(f"[pipeline] Tracked relevant object '{p}'")

    # Process relevant objects
    print("[pipeline] Processing RELEVANT objects first...")
    run_and_union_relevant(relevant_objects, "relevant")
    
    print("[pipeline] Processing ALWAYS_RELEVANT objects...")
    run_and_union_relevant(always_relevant_objects, "always")
    
    # Build per-frame exclusion masks
    exclusion_masks_per_frame = []
    if keep_union is not None:
        exclusion_masks_per_frame = keep_union
        print(f"[pipeline] Built {len(exclusion_masks_per_frame)} per-frame exclusion masks")
    else:
        exclusion_masks_per_frame = [np.zeros((h, w), dtype=np.uint8) for _ in range(total_frames)]
    
    # Process irrelevant objects
    successfully_masked = []
    
    print("[pipeline] Processing IRRELEVANT objects (with per-frame blanking)...")
    
    for i, p in enumerate(irrelevant_objects):
        p_out = os.path.join(out_dir, f"prompt_irrelevant_{i:02d}__{safe_stem(p)}")
        ensure_dir(p_out)
        
        print(f"[pipeline] Detecting irrelevant object '{p}' with blanking")

        masks01 = track_one_prompt_to_masks01(
            repo_root=repo_root,
            video_path=video_path,
            prompt=p,
            out_dir=p_out,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            aot_model=aot_model,
            sam_gap=sam_gap,
            max_obj_num=max_obj_num,
            points_per_side=points_per_side,
            long_term_mem_gap=long_term_mem_gap,
            max_len_long_term=max_len_long_term,
            fps=fps,
            mask_scale=mask_scale,
            debug_prefix=f"irrelevant_",
            use_multiframe_selection=use_multiframe_selection,
            exclusion_masks=exclusion_masks_per_frame,
        )
        
        # Store individual object masks
        object_masks_dict[p] = masks01
        
        # Check if all masks are empty
        if all(m.sum() == 0 for m in masks01):
            print(f"[pipeline] ✗ Object '{p}' failed: all masks empty")
            continue
        
        successfully_masked.append(p)
        
        # Union
        if remove_union is None:
            remove_union = [m.copy() for m in masks01]
        else:
            for t in range(min(len(remove_union), len(masks01))):
                remove_union[t] = np.maximum(remove_union[t], masks01[t])

    # Final dimensions
    L = 0
    H = W = None
    for arr_list in (keep_union, remove_union):
        if arr_list is not None:
            L = len(arr_list)
            H, W = arr_list[0].shape
            break
    if L == 0:
        raise RuntimeError("No objects provided or all prompts empty")

    if keep_union is None:
        keep_union = [np.zeros((H, W), dtype=np.uint8) for _ in range(L)]
    if remove_union is None:
        remove_union = [np.zeros((H, W), dtype=np.uint8) for _ in range(L)]

    # Final: Create composite mask with priority
    # - Base layer: irrelevant objects (WHITE = 1, will be inpainted)
# - Top layer: relevant objects (BLACK = 0, protected from inpainting)
    final = []
    for t in range(L):
        # Start with irrelevant objects (white)
        composite = (remove_union[t] > 0).astype(np.uint8)
        
        # Overlay relevant objects as black (value = 0)
        # This "cuts holes" in the white mask
        relevant_mask = (keep_union[t] > 0)
        composite[relevant_mask] = 0  # ← Black pixels = don't inpaint
        
        final.append(composite)

    return final, object_masks_dict, successfully_masked

def safe_stem(s: str, maxlen: int = 40) -> str:
    s = s.strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_"):
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    if not out:
        out = "obj"
    return out[:maxlen]


# ----------------------------
# main pipeline function
# ----------------------------

def run_samt_mask_pipeline(
    *,
    repo_root: str,
    video_path: str,
    base_out_dir: str,  # 改名：这是基础输出目录
    sample_id: int,      # NEW: sample ID for organizing outputs
    view: str,
    input_type: str = "original",
    stereo_split: str = "side_by_side",
    relevant_objects: list = None,
    irrelevant_objects: list = None,
    always_relevant_objects: list = None,
    mask_scale: float = 1.0,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    aot_model: str = "r50_deaotl",
    sam_gap: int = 9999,
    max_obj_num: int = 2,
    points_per_side: int = 16,
    long_term_mem_gap: int = 9999,
    max_len_long_term: int = 9999,
    fps: int = 0,
    write_per_frame_pngs: bool = True,
) -> dict:
    """
    Produces mask videos for each object and final combined mask.
    Returns dict with validation results.
    
    Args:
        base_out_dir: Base output directory (e.g., /path/to/experiments)
        sample_id: Sample ID (e.g., 0, 1, 2...) for organizing outputs
        
    Outputs will be organized as:
        base_out_dir/
            sample_{sample_id:05d}/
                left/
                right/
                individual_objects/
                final_mask_stereo.mp4
                validation_results.json
                ...
    """

    if view == "wrist":
        relevant_objects.append("black metal stuffs at the bottom of the image")
    else:
        relevant_objects.append("robot arm")

    relevant_objects = relevant_objects or []
    irrelevant_objects = irrelevant_objects or []
    always_relevant_objects = always_relevant_objects or []

    # Create sample-specific output directory
    sample_dir = os.path.join(base_out_dir, f"sample_{sample_id:05d}")
    ensure_dir(sample_dir)
    
    print("\n" + "="*80)
    print(f"SAMPLE {sample_id:05d} - Processing")
    print("="*80)
    print(f"Output directory: {sample_dir}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"View: {view}")
    print(f"Input type: {input_type}")
    print("="*80 + "\n")
    
    repo_root = str(Path(repo_root).resolve())
    video_path = str(Path(video_path).resolve())
    out_dir = str(Path(sample_dir).resolve())

    # View-specific parameters
    if view == "wrist":
        box_threshold = MaskValidationConfig.MIN_CONFIDENCE_WRIST
        text_threshold = MaskValidationConfig.MIN_CONFIDENCE_WRIST
        use_multiframe_selection = True
        print(f"[view=wrist] box_threshold={box_threshold}, KNN multiframe=ON")
    elif view in ["ext1", "ext2"]:
        box_threshold = MaskValidationConfig.MIN_CONFIDENCE_EXT
        text_threshold = MaskValidationConfig.MIN_CONFIDENCE_EXT
        use_multiframe_selection = False
        print(f"[view={view}] Standard mode")
    else:
        raise ValueError(f"Unknown view: {view}")

    fps_eff = fps if fps and fps > 0 else get_fps(video_path, fallback=8)
    print(f"[info] fps={fps_eff}")

    if input_type not in ("original", "stereo"):
        raise ValueError("input_type must be 'original' or 'stereo'")

    if input_type == "original":
        final_masks01, object_masks, successfully_masked = _process_one_video_multiobj(
            repo_root=repo_root,
            video_path=video_path,
            out_dir=out_dir,
            relevant_objects=relevant_objects,
            irrelevant_objects=irrelevant_objects,
            always_relevant_objects=always_relevant_objects,
            mask_scale=mask_scale,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            aot_model=aot_model,
            sam_gap=sam_gap,
            max_obj_num=max_obj_num,
            points_per_side=points_per_side,
            long_term_mem_gap=long_term_mem_gap,
            max_len_long_term=max_len_long_term,
            fps=fps_eff,
            use_multiframe_selection=use_multiframe_selection,
        )

        mask_video_path = os.path.join(out_dir, "final_mask.mp4")
        write_mask_video(final_masks01, mask_video_path, fps_eff)

        if write_per_frame_pngs:
            write_mask_pngs(final_masks01, os.path.join(out_dir, "final_mask_png"))

        return {
            "sample_id": sample_id,
            "sample_dir": out_dir,
            "mask_video": mask_video_path,
            "fps": fps_eff,
            "input_type": input_type,
            "view": view,
            "frames": len(final_masks01),
            "validation_results": {obj: {"status": "success", "type": "relevant" if obj in (relevant_objects + always_relevant_objects) else "irrelevant"} for obj in successfully_masked},
        }

    # Stereo processing
    if stereo_split != "side_by_side":
        raise ValueError("Only stereo_split='side_by_side' supported")

    tmp = os.path.join(out_dir, "_tmp_stereo")
    ensure_dir(tmp)
    left_video = os.path.join(tmp, f"{Path(video_path).stem}_left.mp4")
    right_video = os.path.join(tmp, f"{Path(video_path).stem}_right.mp4")

    crop_stereo_video_ffmpeg(video_path, left_video, "left")
    crop_stereo_video_ffmpeg(video_path, right_video, "right")

    left_out = os.path.join(out_dir, "left")
    right_out = os.path.join(out_dir, "right")
    ensure_dir(left_out)
    ensure_dir(right_out)

    left_masks01, left_object_masks, left_successfully_masked = _process_one_video_multiobj(
        repo_root=repo_root,
        video_path=left_video,
        out_dir=left_out,
        relevant_objects=relevant_objects,
        irrelevant_objects=irrelevant_objects,
        always_relevant_objects=always_relevant_objects,
        mask_scale=mask_scale,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        aot_model=aot_model,
        sam_gap=sam_gap,
        max_obj_num=max_obj_num,
        points_per_side=points_per_side,
        long_term_mem_gap=long_term_mem_gap,
        max_len_long_term=max_len_long_term,
        fps=fps_eff,
        use_multiframe_selection=use_multiframe_selection,
    )

    right_masks01, right_object_masks, right_successfully_masked = _process_one_video_multiobj(
        repo_root=repo_root,
        video_path=right_video,
        out_dir=right_out,
        relevant_objects=relevant_objects,
        irrelevant_objects=irrelevant_objects,
        always_relevant_objects=always_relevant_objects,
        mask_scale=mask_scale,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        aot_model=aot_model,
        sam_gap=sam_gap,
        max_obj_num=max_obj_num,
        points_per_side=points_per_side,
        long_term_mem_gap=long_term_mem_gap,
        max_len_long_term=max_len_long_term,
        fps=fps_eff,
        use_multiframe_selection=use_multiframe_selection,
    )

    # Stereo consistency validation
    print("\n" + "="*60)
    print(f"SAMPLE {sample_id:05d} - STEREO VALIDATION")
    print("="*60)
    
    validation_results = {}
    individual_mask_videos = {}
    
    # Check all objects (relevant + irrelevant)
    all_objects = set(list(relevant_objects) + list(always_relevant_objects) + list(irrelevant_objects))
    
    for obj in all_objects:
        obj_type = "relevant" if obj in (relevant_objects + always_relevant_objects) else "irrelevant"
        
        # Check if object exists in both views
        if obj not in left_object_masks or obj not in right_object_masks:
            validation_results[obj] = {
                "status": "failed",
                "reason": "not_detected_in_both_views",
                "type": obj_type,
            }
            print(f"\n[{obj}] ✗ FAILED: Not detected in both views")
            continue
        
        # Get masks
        left_masks = left_object_masks[obj]
        right_masks = right_object_masks[obj]
        
        # Check if empty
        if all(m.sum() == 0 for m in left_masks) or all(m.sum() == 0 for m in right_masks):
            validation_results[obj] = {
                "status": "failed",
                "reason": "empty_masks",
                "type": obj_type,
            }
            print(f"\n[{obj}] ✗ FAILED: Empty masks")
            continue
        
        # Validate stereo consistency (for irrelevant only)
        if obj_type == "irrelevant":
            passed, metrics = validate_stereo_consistency(left_masks, right_masks, obj)
            
            if passed:
                validation_results[obj] = {
                    "status": "success",
                    "type": obj_type,
                    "metrics": metrics,
                }
            else:
                validation_results[obj] = {
                    "status": "failed",
                    "reason": "stereo_inconsistency",
                    "type": obj_type,
                    "metrics": metrics,
                }
                print(f"\n[{obj}] ✗ FAILED: Stereo inconsistency")
                continue
        else:
            # Relevant objects don't need stereo validation
            validation_results[obj] = {
                "status": "success",
                "type": obj_type,
            }
        
        # Create individual stereo mask video
        print(f"\n[{obj}] ✓ Creating stereo mask video...")
        
        n = min(len(left_masks), len(right_masks))
        stitched_obj = []
        for L, R in zip(left_masks[:n], right_masks[:n]):
            stitched_obj.append(np.concatenate([L, R], axis=1).astype(np.uint8))
        
        # Save individual object video
        individual_dir = os.path.join(out_dir, "individual_objects")
        ensure_dir(individual_dir)
        obj_video_path = os.path.join(individual_dir, f"{safe_stem(obj)}_stereo.mp4")
        write_mask_video(stitched_obj, obj_video_path, fps_eff)
        individual_mask_videos[obj] = obj_video_path
        
        print(f"  Saved: {obj_video_path}")

    # Create final combined mask
    n = min(len(left_masks01), len(right_masks01))
    stitched_final = []
    for L, R in zip(left_masks01[:n], right_masks01[:n]):
        stitched_final.append(np.concatenate([L, R], axis=1).astype(np.uint8))

    mask_video_path = os.path.join(out_dir, "final_mask_stereo.mp4")
    write_mask_video(stitched_final, mask_video_path, fps_eff)

    if write_per_frame_pngs:
        write_mask_pngs(stitched_final, os.path.join(out_dir, "final_mask_png"))

    # Save validation results to JSON
    validation_json_path = os.path.join(out_dir, "validation_results.json")
    
    # Add metadata to validation results
    validation_output = {
        "sample_id": sample_id,
        "video_path": video_path,
        "view": view,
        "input_type": input_type,
        "fps": fps_eff,
        "total_frames": len(stitched_final),
        "relevant_objects": relevant_objects,
        "irrelevant_objects": irrelevant_objects,
        "always_relevant_objects": always_relevant_objects,
        "validation_results": validation_results,
    }
    
    with open(validation_json_path, 'w') as f:
        json.dump(validation_output, f, indent=2)
    
    print("\n" + "="*60)
    print(f"SAMPLE {sample_id:05d} - VALIDATION SUMMARY")
    print("="*60)
    successful = [obj for obj, res in validation_results.items() if res["status"] == "success"]
    failed = [obj for obj, res in validation_results.items() if res["status"] == "failed"]
    
    print(f"Successful: {len(successful)}/{len(validation_results)}")
    for obj in successful:
        print(f"  ✓ {obj} ({validation_results[obj]['type']})")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(validation_results)}")
        for obj in failed:
            reason = validation_results[obj].get('reason', 'unknown')
            print(f"  ✗ {obj} ({validation_results[obj]['type']}): {reason}")
    
    print(f"\nValidation results saved to: {validation_json_path}")
    print("="*60)

    return {
        "sample_id": sample_id,
        "sample_dir": out_dir,
        "mask_video": mask_video_path,
        "fps": fps_eff,
        "input_type": input_type,
        "view": view,
        "frames": len(stitched_final),
        "left_dir": left_out,
        "right_dir": right_out,
        "validation_results": validation_results,
        "individual_mask_videos": individual_mask_videos,
        "validation_json": validation_json_path,
    }


if __name__ == "__main__":
    # Example: process single sample
    res = run_samt_mask_pipeline(
        repo_root="/viscam/u/jeffreywang0303/projects/Inpainting-Objects/Segment-and-Track-Anything",
        video_path="/viscam/data/DROID/droid_raw_t/1.0.1/CLVR/success/2023-06-13/Tue_Jun_13_01:40:39_2023/recordings/MP4/16787047-stereo.mp4",
        base_out_dir="/viscam/u/jeffreywang0303/projects/Inpainting-Objects/experiments/E7",  # Base directory
        sample_id=0,  # Sample ID
        
        view="wrist",
        input_type="stereo",
        mask_scale=1.01,

        relevant_objects=["white mug"],
        always_relevant_objects=["purple mat"],
        irrelevant_objects=["white tube"],

        sam_gap=9999,
        max_obj_num=1,
    )
    
    print("\n" + "="*80)
    print(f"SAMPLE {res['sample_id']:05d} - FINAL RESULTS")
    print("="*80)
    print(f"Sample directory: {res['sample_dir']}")
    print(f"Final mask video: {res['mask_video']}")
    print(f"Validation JSON: {res['validation_json']}")
    print(f"\nIndividual object videos:")
    for obj, path in res['individual_mask_videos'].items():
        status = res['validation_results'][obj]['status']
        print(f"  [{status.upper()}] {obj}: {path}")
    print("="*80)
    