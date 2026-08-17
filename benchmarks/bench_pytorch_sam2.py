#!/usr/bin/env python3
"""PyTorch official SAM2.1 video-tracking benchmark, mirrored against
sam3.cpp's sam3_benchmark pipeline (encode frame 0 + point prompt, then
propagate across frames). Reports per-frame latency on the same GPU.

Usage:
    uv run python benchmarks/bench_pytorch_sam2.py --frames 3 --model sam2.1_hiera_tiny
"""
import argparse
import os
import subprocess
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

MODEL_ID = {
    "sam2.1_hiera_tiny": ("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt"),
    "sam2.1_hiera_small": ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt"),
    "sam2.1_hiera_base_plus": ("facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt"),
    "sam2.1_hiera_large": ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"),
}


def load_video_frames(video_path, n_frames):
    """Extract n frames from a video as [H, W, 3] uint8 via ffmpeg."""
    frames = []
    for i in range(n_frames):
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"select=eq(n\\,{i})", "-vsync", "vfr", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        raw = subprocess.run(cmd, capture_output=True).stdout
        if not raw:
            raise RuntimeError(f"ffmpeg frame {i} failed")
        # 1920x1080 is known for the bundled test_video.mp4
        h, w = 1080, 1920
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
    return frames


def bench(model_name, video_path, n_frames, img_size):
    from sam2.build_sam import build_sam2_video_predictor_hf

    repo_id, _ = MODEL_ID[model_name]

    torch.cuda.synchronize()
    # Downloads the checkpoint + config from HF on first use. image_size
    # mirrors sam3.cpp's default encode size so timings are comparable.
    predictor = build_sam2_video_predictor_hf(repo_id, image_size=img_size)
    predictor = predictor.to("cuda")
    predictor.eval()

    torch.cuda.synchronize()

    # Frame 0: init state + point prompt + propagate (== C++ init phase)
    t0 = time.perf_counter()
    with torch.inference_mode():
        state = predictor.init_state(video_path)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=0,
            points=np.array([[315.0, 250.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        for _ in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=1):
            pass
        torch.cuda.synchronize()
    t_frame0 = time.perf_counter() - t0

    # Frames 1..n-1: memory propagation (== C++ track phase)
    t_track_sum = 0.0
    for f in range(1, n_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in predictor.propagate_in_video(state, start_frame_idx=f, max_frame_num_to_track=1):
                pass
            torch.cuda.synchronize()
        t_track_sum += time.perf_counter() - t0

    print(f"\nPyTorch {model_name} (img_size={img_size}, {n_frames} frames, "
          f"{torch.cuda.get_device_name(0)}):")
    print(f"  frame0 (encode+prompt+propagate): {t_frame0 * 1000:.1f} ms")
    if n_frames > 1:
        print(f"  track/frame (propagate):          {t_track_sum / (n_frames - 1) * 1000:.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam2.1_hiera_tiny", choices=list(MODEL_ID))
    ap.add_argument("--video", default="data/test_video.mp4")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=1008)
    args = ap.parse_args()
    bench(args.model, args.video, args.frames, args.img_size)


if __name__ == "__main__":
    main()
