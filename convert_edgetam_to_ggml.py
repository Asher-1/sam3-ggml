#!/usr/bin/env python3
"""Convert EdgeTAM PyTorch checkpoint to ggml binary format.

EdgeTAM uses a RepViT-M1 backbone with BatchNorm and RepVGG-style
reparameterization. This script:
  1. Fuses all BatchNorm layers into preceding convolutions
  2. Reparameterizes RepVGG token mixers (3-branch → single DW 3×3)
  3. Maps tensor names to ggml format
  4. Writes a SAM2-compatible binary with backbone_type=2

Usage:
    uv run python convert_edgetam_to_ggml.py \
        --model ~/Documents/EdgeTAM/checkpoints/edgetam.pt \
        --output models/edgetam_f16.ggml \
        --ftype 1
"""

import argparse
import struct
import sys
import os
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

MAGIC   = 0x73616D32   # "sam2" — reuse SAM2 magic, dispatch on backbone_type
VERSION = 1
FTYPE_F32 = 0
FTYPE_F16 = 1

# EdgeTAM hyperparameters (from edgetam.yaml)
EDGETAM_HPARAMS = {
    "image_size":              1024,
    "backbone_type":           2,     # 2 = repvit (EdgeTAM)

    # Hiera fields (dummy — ignored when backbone_type == 2)
    "hiera_embed_dim":         0,
    "hiera_num_heads":         0,
    "hiera_num_stages":        0,
    "hiera_stages":            [0, 0, 0, 0],
    "hiera_global_att_n":      0,
    "hiera_global_att_idx":    [0, 0, 0, 0, 0, 0, 0, 0],
    "hiera_q_pool":            0,
    "hiera_window_spec":       [0, 0, 0, 0],
    "hiera_pos_embed_bkg_h":   0,
    "hiera_pos_embed_bkg_w":   0,
    "scalp":                   1,

    # FPN neck
    "neck_dim":                256,
    "fpn_top_down_levels_n":   2,
    "fpn_top_down_levels":     [2, 3],

    # SAM decoder
    "sam_embed_dim":           256,
    "sam_dec_depth":           2,
    "sam_n_multimask":         3,
    "sam_iou_head_depth":      3,

    # Memory
    "mem_out_dim":             64,
    "mem_attn_layers":         2,
    "num_maskmem":             7,
    "max_obj_ptrs":            16,

    # Sigmoid
    "sigmoid_scale_for_mem_enc_x100":  2000,
    "sigmoid_bias_for_mem_enc_x100":   -1000,

    # Boolean flags
    "use_high_res_features":               1,
    "use_obj_ptrs_in_encoder":             1,
    "pred_obj_scores":                     1,
    "use_multimask_token_for_obj_ptr":     1,
    "directly_add_no_mem_embed":           1,
    "non_overlap_masks_for_mem_enc":       1,
    "binarize_mask_from_pts":              0,
    "multimask_output_for_tracking":       1,
    "multimask_min_pt_num":                0,
    "multimask_max_pt_num":                1,
    "fixed_no_obj_ptr":                    1,
    "iou_prediction_use_sigmoid":          1,
    "use_mask_input_as_output":            1,
    "multimask_output_in_sam":             1,
    "is_sam2_1":                           0,

    # EdgeTAM-specific
    "repvit_num_stages":       4,
    "repvit_stages":           [2, 2, 14, 2],
    "repvit_channels":         [48, 96, 192, 384],
    "repvit_se_ratio_x100":    25,
    "has_spatial_perceiver":   1,
    "perceiver_depth":         2,
    "perceiver_dim":           64,
    "perceiver_num_latents_1d": 256,
    "perceiver_num_latents_2d": 256,
    "perceiver_ff_mult":       4,
    "mem_attn_ca_type":        1,  # 0=RoPEv1, 1=RoPEv2
    "mem_attn_ca_q_size":      64,
    "mem_attn_ca_k_size":      16,
}


# ── BN Fusion ──────────────────────────────────────────────────────────────────

def fuse_bn_into_conv(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d with weight and bias."""
    import torch
    bn_std = (bn_var + eps).sqrt()
    scale = bn_weight / bn_std

    if conv_weight.dim() == 4:
        fused_weight = conv_weight * scale.view(-1, 1, 1, 1)
    else:
        fused_weight = conv_weight * scale

    if conv_bias is not None:
        fused_bias = (conv_bias - bn_mean) * scale + bn_bias
    else:
        fused_bias = -bn_mean * scale + bn_bias

    return fused_weight, fused_bias


def repvgg_reparameterize(sd, prefix, ch, eps=1e-5):
    """Fuse RepVGG 3-way parallel branches + optional final BN into single DW 3×3.

    Branches:
      1. DW Conv 3×3 + BN  (prefix.conv.c.weight + prefix.conv.bn.*)
      2. Conv 1×1 + BN     (prefix.conv1.c.weight + prefix.conv1.bn.*)
      3. Identity           (implicit)
    Final BN: prefix.bn.weight/bias/running_mean/running_var (may be nn.Identity → absent)

    Returns (fused_weight, fused_bias) as numpy arrays.
    """
    import torch
    import torch.nn.functional as F

    # Branch 1: DW 3×3 + BN
    dw_w = sd[f"{prefix}.conv.c.weight"]
    dw_bn = (sd[f"{prefix}.conv.bn.weight"],
             sd[f"{prefix}.conv.bn.bias"],
             sd[f"{prefix}.conv.bn.running_mean"],
             sd[f"{prefix}.conv.bn.running_var"])
    dw_fused_w, dw_fused_b = fuse_bn_into_conv(dw_w, None, *dw_bn, eps=eps)

    # Branch 2: Conv 1×1 + BN
    c1_w = sd[f"{prefix}.conv1.c.weight"]
    c1_bn = (sd[f"{prefix}.conv1.bn.weight"],
             sd[f"{prefix}.conv1.bn.bias"],
             sd[f"{prefix}.conv1.bn.running_mean"],
             sd[f"{prefix}.conv1.bn.running_var"])
    c1_fused_w, c1_fused_b = fuse_bn_into_conv(c1_w, None, *c1_bn, eps=eps)

    # Pad 1×1 to 3×3: (ch, 1, 1, 1) → (ch, 1, 3, 3)
    c1_padded = F.pad(c1_fused_w, [1, 1, 1, 1])

    # Branch 3: Identity kernel — center pixel = 1
    id_kernel = torch.zeros_like(dw_fused_w)  # (ch, 1, 3, 3)
    id_kernel[:, 0, 1, 1] = 1.0

    # Sum all three branches
    merged_w = dw_fused_w + c1_padded + id_kernel
    merged_b = dw_fused_b + c1_fused_b  # identity has zero bias

    # Final BN: check if it exists (legacy mode has nn.Identity → no weight key)
    final_bn_key = f"{prefix}.bn.weight"
    if final_bn_key in sd:
        final_bn = (sd[f"{prefix}.bn.weight"],
                     sd[f"{prefix}.bn.bias"],
                     sd[f"{prefix}.bn.running_mean"],
                     sd[f"{prefix}.bn.running_var"])
        merged_w, merged_b = fuse_bn_into_conv(merged_w, merged_b, *final_bn, eps=eps)

    return merged_w.float().numpy(), merged_b.float().numpy()


def fuse_conv_bn(sd, conv_prefix, bn_prefix, eps=1e-5):
    """Fuse a standard Conv + BN pair. Returns (weight, bias) as numpy arrays."""
    conv_w = sd[f"{conv_prefix}.weight"]
    conv_b = sd.get(f"{conv_prefix}.bias", None)
    bn_w = sd[f"{bn_prefix}.weight"]
    bn_b = sd[f"{bn_prefix}.bias"]
    bn_mean = sd[f"{bn_prefix}.running_mean"]
    bn_var = sd[f"{bn_prefix}.running_var"]
    w, b = fuse_bn_into_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=eps)
    return w.float().numpy(), b.float().numpy()


# ── Backbone conversion ───────────────────────────────────────────────────────

def convert_repvit_backbone(sd):
    """Convert RepViT backbone tensors with BN fusion and RepVGG reparameterization.

    Returns dict of {ggml_name: numpy_array}.
    """
    out = {}

    # Stem: conv1 + BN → GELU → conv2 + BN (no activation)
    w, b = fuse_conv_bn(sd,
                         "image_encoder.trunk.body.stem.conv1.c",
                         "image_encoder.trunk.body.stem.conv1.bn")
    out["repvit.stem.conv1.weight"] = w
    out["repvit.stem.conv1.bias"] = b

    w, b = fuse_conv_bn(sd,
                         "image_encoder.trunk.body.stem.conv2.c",
                         "image_encoder.trunk.body.stem.conv2.bn")
    out["repvit.stem.conv2.weight"] = w
    out["repvit.stem.conv2.bias"] = b

    stages = [2, 2, 14, 2]
    channels = [48, 96, 192, 384]

    for s in range(4):
        n_blocks = stages[s]
        ch = channels[s]
        src_prefix = f"image_encoder.trunk.body.stages_{s}"

        # Stage blocks
        for b_idx in range(n_blocks):
            src_blk = f"{src_prefix}.blocks.{b_idx}"
            dst_blk = f"repvit.stages.{s}.blocks.{b_idx}"

            # Token mixer: RepVGG reparameterization
            w, bias = repvgg_reparameterize(sd, f"{src_blk}.token_mixer", ch)
            out[f"{dst_blk}.tm.weight"] = w
            out[f"{dst_blk}.tm.bias"] = bias

            # SE (only on even-indexed blocks)
            se_key = f"{src_blk}.se.fc1.weight"
            if se_key in sd:
                out[f"{dst_blk}.se.fc1.weight"] = sd[f"{src_blk}.se.fc1.weight"].float().numpy()
                out[f"{dst_blk}.se.fc1.bias"] = sd[f"{src_blk}.se.fc1.bias"].float().numpy()
                out[f"{dst_blk}.se.fc2.weight"] = sd[f"{src_blk}.se.fc2.weight"].float().numpy()
                out[f"{dst_blk}.se.fc2.bias"] = sd[f"{src_blk}.se.fc2.bias"].float().numpy()

            # Channel mixer: conv1 + BN, conv2 + BN
            w, bias = fuse_conv_bn(sd, f"{src_blk}.channel_mixer.conv1.c",
                                       f"{src_blk}.channel_mixer.conv1.bn")
            out[f"{dst_blk}.cm.conv1.weight"] = w
            out[f"{dst_blk}.cm.conv1.bias"] = bias

            w, bias = fuse_conv_bn(sd, f"{src_blk}.channel_mixer.conv2.c",
                                       f"{src_blk}.channel_mixer.conv2.bn")
            out[f"{dst_blk}.cm.conv2.weight"] = w
            out[f"{dst_blk}.cm.conv2.bias"] = bias

        # Downsample (stages 1, 2, 3 have downsamples at the start)
        ds_key = f"{src_prefix}.downsample.spatial_downsample.c.weight"
        if ds_key in sd:
            ds_src = f"{src_prefix}.downsample"
            ds_dst = f"repvit.stages.{s}.ds"

            # Pre-block token mixer (RepVGG reparameterized)
            prev_ch = channels[s - 1] if s > 0 else channels[0]
            w, bias = repvgg_reparameterize(sd, f"{ds_src}.pre_block.token_mixer", prev_ch)
            out[f"{ds_dst}.pre.tm.weight"] = w
            out[f"{ds_dst}.pre.tm.bias"] = bias

            # Pre-block channel mixer
            w, bias = fuse_conv_bn(sd, f"{ds_src}.pre_block.channel_mixer.conv1.c",
                                       f"{ds_src}.pre_block.channel_mixer.conv1.bn")
            out[f"{ds_dst}.pre.cm.conv1.weight"] = w
            out[f"{ds_dst}.pre.cm.conv1.bias"] = bias

            w, bias = fuse_conv_bn(sd, f"{ds_src}.pre_block.channel_mixer.conv2.c",
                                       f"{ds_src}.pre_block.channel_mixer.conv2.bn")
            out[f"{ds_dst}.pre.cm.conv2.weight"] = w
            out[f"{ds_dst}.pre.cm.conv2.bias"] = bias

            # Spatial downsample (DW conv + BN)
            w, bias = fuse_conv_bn(sd, f"{ds_src}.spatial_downsample.c",
                                       f"{ds_src}.spatial_downsample.bn")
            out[f"{ds_dst}.spatial.weight"] = w
            out[f"{ds_dst}.spatial.bias"] = bias

            # Channel downsample (1×1 conv + BN)
            w, bias = fuse_conv_bn(sd, f"{ds_src}.channel_downsample.c",
                                       f"{ds_src}.channel_downsample.bn")
            out[f"{ds_dst}.channel.weight"] = w
            out[f"{ds_dst}.channel.bias"] = bias

            # FFN (conv1 + BN, conv2 + BN)
            w, bias = fuse_conv_bn(sd, f"{ds_src}.ffn.conv1.c",
                                       f"{ds_src}.ffn.conv1.bn")
            out[f"{ds_dst}.ffn.conv1.weight"] = w
            out[f"{ds_dst}.ffn.conv1.bias"] = bias

            w, bias = fuse_conv_bn(sd, f"{ds_src}.ffn.conv2.c",
                                       f"{ds_src}.ffn.conv2.bn")
            out[f"{ds_dst}.ffn.conv2.weight"] = w
            out[f"{ds_dst}.ffn.conv2.bias"] = bias

    return out


# ── Shared component renaming (reuse SAM2 patterns) ──────────────────────────

def rename_shared_key(k):
    """Map shared component keys (FPN, SAM, memory) to ggml names.

    Returns None if the tensor should be skipped.
    """
    # Skip patterns
    skip_patterns = ["loss", "criterion", "_dn_", "label_enc",
                     "num_batches_tracked", "running_mean", "running_var",
                     "image_encoder.trunk."]
    for pat in skip_patterns:
        if pat in k:
            return None

    # FPN Neck
    k = k.replace("image_encoder.neck.convs.", "fpn.convs.")
    k = k.replace(".conv.weight", ".weight")
    k = k.replace(".conv.bias", ".bias")

    # SAM prompt encoder
    k = k.replace("sam_prompt_encoder.", "sam_pe.")
    k = k.replace("sam_pe.pe_layer.positional_encoding_gaussian_matrix",
                   "sam_pe.pe_gaussian")
    k = k.replace("sam_pe.mask_downscaling.", "sam_pe.mask_ds.")

    # SAM mask decoder
    k = k.replace("sam_mask_decoder.", "sam_dec.")
    k = k.replace("sam_dec.transformer.layers.", "sam_dec.twoway.")
    k = k.replace("sam_dec.transformer.final_attn_token_to_image.",
                   "sam_dec.final_attn.")
    k = k.replace("sam_dec.transformer.norm_final_attn.",
                   "sam_dec.final_norm.")
    k = k.replace("sam_dec.output_upscaling.", "sam_dec.upscale.")
    k = k.replace("sam_dec.output_hypernetworks_mlps.", "sam_dec.hyper.")

    # Self-attention renaming (global)
    k = k.replace(".self_attn.", ".sa.")

    # TwoWay MLP layers
    if "sam_dec.twoway." in k and ".mlp." in k:
        k = k.replace(".mlp.layers.0.", ".mlp.lin1.")
        k = k.replace(".mlp.layers.1.", ".mlp.lin2.")

    # Memory encoder
    k = k.replace("memory_encoder.", "mem_enc.")
    k = k.replace("mem_enc.mask_downsampler.encoder.", "mem_enc.ds.")
    k = k.replace("mem_enc.fuser.layers.", "mem_enc.fuser.")

    # Memory attention
    k = k.replace("memory_attention.layers.", "mem_attn.layers.")
    k = k.replace("memory_attention.norm.", "mem_attn.norm.")
    k = k.replace(".cross_attn_image.", ".ca.")

    # Top-level
    k = k.replace("maskmem_tpos_enc", "mem_enc.tpos_enc")
    k = k.replace("mask_downsample.", "trk_mask_ds.")

    # Spatial perceiver
    k = k.replace("spatial_perceiver.latents_2d", "perceiver.latents_2d")
    k = k.replace("spatial_perceiver.latents", "perceiver.latents")
    k = k.replace("spatial_perceiver.norm.", "perceiver.norm.")
    k = k.replace("spatial_perceiver.layers.", "perceiver.layers.")
    # Perceiver sub-module renaming
    k = k.replace(".attn.norm_latents.", ".ca.norm_latents.")
    k = k.replace(".attn.norm_x.", ".ca.norm_x.")
    k = k.replace(".attn.to_q.", ".ca.q.")
    k = k.replace(".attn.to_kv.", ".ca.kv.")
    k = k.replace(".attn.to_out.", ".ca.out.")
    # FFN after cross-attention: .ff.0 → .ff.norm, .ff.1 → .ff.fc1, .ff.3 → .ff.fc2
    k = k.replace(".ff.0.", ".ff.norm.")
    k = k.replace(".ff.1.", ".ff.fc1.")
    k = k.replace(".ff.3.", ".ff.fc2.")
    # Self-attention in perceiver (already handled by .sa. above for .self_attn.)
    k = k.replace(".sa.norm.", ".sa.norm.")
    k = k.replace(".sa.to_q.", ".sa.q.")
    k = k.replace(".sa.to_kv.", ".sa.kv.")
    k = k.replace(".sa.to_out.", ".sa.out.")
    # Self-FFN: .self_ff.0 → .sa_ff.norm, .self_ff.1 → .sa_ff.fc1, .self_ff.3 → .sa_ff.fc2
    k = k.replace(".self_ff.0.", ".sa_ff.norm.")
    k = k.replace(".self_ff.1.", ".sa_ff.fc1.")
    k = k.replace(".self_ff.3.", ".sa_ff.fc2.")

    return k


# ── GGUF output ─────────────────────────────────────────────────────────────

# Map this script's hparams dict keys to the loader's sam3.hparams.* KV names.
# Keys absent from the map are written verbatim.
HPARAMS_KEY_MAP = {
    "image_size":                     "img_size",
    "hiera_global_att_n":             "hiera_global_n",
    "hiera_global_att_idx":           "hiera_global_idx",
    "fpn_top_down_levels_n":          "fpn_top_down_n",
    "sigmoid_scale_for_mem_enc_x100": "sigmoid_scale_x100",
    "sigmoid_bias_for_mem_enc_x100":  "sigmoid_bias_x100",
    "has_spatial_perceiver":          "has_perceiver",
    "perceiver_num_latents_1d":       "perceiver_n_latents_1d",
    "perceiver_num_latents_2d":       "perceiver_n_latents_2d",
}

# Array hparams written as INT32 arrays: (dict key, max length)
HPARAMS_ARRAYS = [
    ("hiera_stages", 4),
    ("hiera_global_att_idx", 8),
    ("hiera_window_spec", 4),
    ("fpn_top_down_levels", 4),
    ("repvit_stages", 4),
    ("repvit_channels", 4),
]

# Write the model as a standard GGUF v3 file (see scripts/gguf_writer.py).
# EdgeTAM models use arch "edgetam" (backbone_type == 2 in the hparams).
def write_gguf(path: str, ftype: int, renamed: dict, hp: dict):
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scripts"))
    from gguf_writer import GGUFWriter, GGML_TYPE_F16, GGML_TYPE_F32

    w = GGUFWriter()
    w.add_str("general.architecture", "edgetam")
    w.add_str("sam3.arch", "edgetam")
    w.add_i32("sam3.version", VERSION)
    w.add_i32("sam3.ftype", ftype)

    # Hparams → KV (array fields merged into INT32 arrays)
    for key, val in hp.items():
        if any(key == ak for ak, _ in HPARAMS_ARRAYS):
            continue   # handled below
        w.add_i32(f"sam3.hparams.{HPARAMS_KEY_MAP.get(key, key)}", val)
    for key, max_len in HPARAMS_ARRAYS:
        arr = list(hp.get(key, []))
        arr = (arr + [0] * max_len)[:max_len]
        w.add_arr_i32(f"sam3.hparams.{HPARAMS_KEY_MAP.get(key, key)}", arr)

    # Tensors (ggml column-major ne = reversed PyTorch shape)
    for name, data in renamed.items():
        ne = list(reversed(data.shape))
        use_f16 = (ftype == FTYPE_F16 and len(data.shape) >= 2
                   and "embed" not in name
                   and "pos_embed" not in name
                   and "tpos" not in name
                   and "pe_gaussian" not in name
                   and "token" not in name
                   and "no_obj" not in name
                   and "no_mem" not in name
                   and "gamma" not in name
                   and "latents" not in name)
        ttype = GGML_TYPE_F16 if use_f16 else GGML_TYPE_F32
        dt = data.astype(np.float16 if use_f16 else np.float32)
        w.add_tensor(name, ne, ttype, dt.tobytes())

    with open(path, "wb") as fout:
        w.write_header_and_meta(fout)
        for i in range(len(renamed)):
            w.write_tensor_data(fout, i)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert EdgeTAM checkpoint to ggml format")
    parser.add_argument("--model", required=True, help="Path to edgetam.pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .ggml file path")
    parser.add_argument("--ftype", type=int, default=1, choices=[0, 1],
                        help="0=f32, 1=f16 (default)")
    args = parser.parse_args()

    import torch

    print(f"Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)

    if "model" in checkpoint:
        sd = checkpoint["model"]
    elif "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
    else:
        sd = checkpoint

    print(f"  {len(sd)} tensors in checkpoint")

    hp = dict(EDGETAM_HPARAMS)

    # ── Convert backbone (BN fusion + RepVGG reparameterization) ──────
    print("Fusing BatchNorm + RepVGG reparameterization...")
    backbone_tensors = convert_repvit_backbone(sd)
    print(f"  RepViT backbone: {len(backbone_tensors)} fused tensors")

    # ── Convert shared components ─────────────────────────────────────
    shared_tensors = {}
    skipped = []
    for k, v in sd.items():
        # Skip backbone tensors (already handled above)
        if "image_encoder.trunk." in k:
            continue

        new_name = rename_shared_key(k)
        if new_name is None:
            skipped.append(k)
            continue
        data = v.detach().float().numpy()
        shared_tensors[new_name] = data

    print(f"  Shared components: {len(shared_tensors)} tensors")
    if skipped:
        print(f"  Skipped: {len(skipped)} tensors")

    # ── Merge all tensors ────────────────────────────────────────────
    all_tensors = {}
    all_tensors.update(backbone_tensors)
    all_tensors.update(shared_tensors)

    # ── Print inventory ──────────────────────────────────────────────
    n_repvit = sum(1 for k in all_tensors if k.startswith("repvit."))
    n_fpn = sum(1 for k in all_tensors if k.startswith("fpn."))
    n_perceiver = sum(1 for k in all_tensors if k.startswith("perceiver."))
    n_sam_pe = sum(1 for k in all_tensors if k.startswith("sam_pe."))
    n_sam_dec = sum(1 for k in all_tensors if k.startswith("sam_dec."))
    n_mem_enc = sum(1 for k in all_tensors if k.startswith("mem_enc."))
    n_mem_attn = sum(1 for k in all_tensors if k.startswith("mem_attn."))
    n_other = len(all_tensors) - n_repvit - n_fpn - n_perceiver - n_sam_pe - n_sam_dec - n_mem_enc - n_mem_attn
    print(f"\n  RepViT={n_repvit}, FPN={n_fpn}, Perceiver={n_perceiver}, "
          f"SAM_PE={n_sam_pe}, SAM_DEC={n_sam_dec}, "
          f"MEM_ENC={n_mem_enc}, MEM_ATTN={n_mem_attn}, other={n_other}")
    print(f"  Total: {len(all_tensors)} tensors")

    # ── Write output ─────────────────────────────────────────────────
    print(f"\nWriting: {args.output}")
    write_gguf(args.output, args.ftype, all_tensors, hp)

    file_size = os.path.getsize(args.output)
    print(f"Done. {len(all_tensors)} tensors, {file_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
