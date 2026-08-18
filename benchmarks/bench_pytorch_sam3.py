#!/usr/bin/env python3
"""Benchmark Meta's official SAM 3 point-prompt path on a local CUDA GPU.

The default loads facebook/sam3 from Hugging Face and therefore requires
checkpoint access. Use --random-weights only to measure the official graph's
latency when checkpoint access is unavailable; those outputs are not valid for
accuracy comparisons.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-repo",
        help="Path to a checkout of https://github.com/facebookresearch/sam3",
    )
    parser.add_argument("--checkpoint", help="Path to the official sam3.pt checkpoint")
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Skip checkpoint loading; measures latency only, not accuracy",
    )
    parser.add_argument("--image", default=str(repo_root / "tests" / "cat.jpg"))
    parser.add_argument("--point", nargs=2, type=float, default=(315.0, 250.0))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    if args.random_weights and args.checkpoint:
        parser.error("--random-weights and --checkpoint are mutually exclusive")
    return args


def timed_cuda_call(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def main():
    args = parse_args()
    if args.official_repo:
        sys.path.insert(0, str(Path(args.official_repo).resolve()))

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    image = Image.open(args.image).convert("RGB")
    point = np.asarray([args.point], dtype=np.float32)
    label = np.asarray([1], dtype=np.int32)
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    load_from_hf = args.checkpoint is None and not args.random_weights

    build_start = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=load_from_hf,
        enable_inst_interactivity=True,
        compile=args.compile,
    )
    processor = Sam3Processor(model)
    torch.cuda.synchronize()

    if args.random_weights:
        weight_source = "random weights (latency only)"
    elif args.checkpoint:
        weight_source = str(Path(args.checkpoint).resolve())
    else:
        weight_source = "facebook/sam3 from Hugging Face"

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"weights: {weight_source}")
    print(f"dtype: {args.dtype}, compile: {args.compile}")
    print(f"build_ms: {(time.perf_counter() - build_start) * 1000.0:.3f}")

    encode_times = []
    segment_times = []
    total_runs = args.warmup + args.repeat
    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        for index in range(total_runs):
            state, encode_ms = timed_cuda_call(lambda: processor.set_image(image))
            _, segment_ms = timed_cuda_call(
                lambda: model.predict_inst(
                    state,
                    point_coords=point,
                    point_labels=label,
                    multimask_output=False,
                )
            )
            is_warmup = index < args.warmup
            sample_index = index + 1 if is_warmup else index - args.warmup + 1
            phase = "warmup" if is_warmup else "timed"
            print(
                f"{phase} {sample_index}: encode_ms={encode_ms:.3f} "
                f"segment_ms={segment_ms:.3f}"
            )
            if not is_warmup:
                encode_times.append(encode_ms)
                segment_times.append(segment_ms)

    encode_avg = sum(encode_times) / len(encode_times)
    segment_avg = sum(segment_times) / len(segment_times)
    print(f"encode_avg_ms: {encode_avg:.3f}")
    print(f"segment_avg_ms: {segment_avg:.3f}")
    print(f"inference_avg_ms: {encode_avg + segment_avg:.3f}")
    encode_p50 = statistics.median(encode_times)
    segment_p50 = statistics.median(segment_times)
    print(f"encode_p50_ms: {encode_p50:.3f}")
    print(f"segment_p50_ms: {segment_p50:.3f}")
    print(f"inference_p50_ms: {encode_p50 + segment_p50:.3f}")
    print(f"peak_vram_mib: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
