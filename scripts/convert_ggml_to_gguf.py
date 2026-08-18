#!/usr/bin/env python3
"""Repack the legacy .ggml weight files into the standard GGUF format.

The .ggml format is a custom container used only by sam3.cpp (magic "sam3"
or "sam2" followed by positional int32 hparams and streamed tensor records).
Every other ggml-based project in this ecosystem (face-detect-ggml,
free-splatter.cpp, OpenPCDet-GGML) loads weights through gguf_init_from_file
+ gguf KV lookups + ggml_get_tensor. This script migrates existing .ggml
files to that shared standard — no PyTorch checkpoints needed, because the
tensor dtype values in .ggml are already GGML_TYPE enum values and the data
blocks are byte-identical to what GGUF stores.

Usage:
    python convert_ggml_to_gguf.py <model.ggml> [out.gguf]
    python convert_ggml_to_gguf.py --dir models/        # repack everything
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from gguf_writer import (  # noqa: E402
    GGUFWriter,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q8_0,
    ggml_nbytes,
)

SAM3_MAGIC = 0x73616D33   # "sam3"
SAM2_MAGIC = 0x73616D32   # "sam2"
TOK_MAGIC  = 0x746F6B00   # "tok\0"

ALIGN = 32

# Per-record: (ggml name, [count]) where count is how many int32 elements are
# consumed for array fields. Order MUST match sam3_load_hparams() in sam3.cpp.
SAM3_HPARAMS = [
    ("img_size", 1), ("patch_size", 1), ("vit_embed_dim", 1), ("vit_depth", 1),
    ("vit_num_heads", 1), ("vit_mlp_ratio_x1000", 1), ("vit_window_size", 1),
    ("n_global_attn", 1), ("global_attn_idx", 4),
    ("text_width", 1), ("text_heads", 1), ("text_layers", 1), ("text_ctx_len", 1),
    ("text_vocab_size", 1), ("text_out_dim", 1), ("neck_dim", 1),
    ("fenc_layers", 1), ("fenc_heads", 1), ("fenc_ffn_dim", 1),
    ("ddec_layers", 1), ("ddec_heads", 1), ("ddec_ffn_dim", 1),
    ("ddec_num_queries", 1), ("geom_layers", 1), ("n_presence_tokens", 1),
    ("n_geom_queries", 1), ("sam_embed_dim", 1), ("sam_dec_depth", 1),
    ("sam_n_multimask", 1), ("sam_iou_head_depth", 1), ("mem_out_dim", 1),
    ("mem_attn_layers", 1), ("num_maskmem", 1), ("max_obj_ptrs", 1),
    ("n_amb_experts", 1), ("visual_only", 1),
]

# Order MUST match sam2_load_hparams() in sam3.cpp.
SAM2_HPARAMS = [
    ("img_size", 1), ("backbone_type", 1),
    ("hiera_embed_dim", 1), ("hiera_num_heads", 1), ("hiera_num_stages", 1),
    ("hiera_stages", 4), ("hiera_global_n", 1), ("hiera_global_idx", 8),
    ("hiera_q_pool", 1), ("hiera_window_spec", 4),
    ("hiera_pos_embed_bkg_h", 1), ("hiera_pos_embed_bkg_w", 1), ("scalp", 1),
    ("neck_dim", 1), ("fpn_top_down_n", 1), ("fpn_top_down_levels", 4),
    ("sam_embed_dim", 1), ("sam_dec_depth", 1), ("sam_n_multimask", 1),
    ("sam_iou_head_depth", 1), ("mem_out_dim", 1), ("mem_attn_layers", 1),
    ("num_maskmem", 1), ("max_obj_ptrs", 1),
    ("sigmoid_scale_x100", 1), ("sigmoid_bias_x100", 1),
    ("use_high_res_features", 1), ("use_obj_ptrs_in_encoder", 1),
    ("pred_obj_scores", 1), ("use_multimask_token_for_obj_ptr", 1),
    ("directly_add_no_mem_embed", 1), ("non_overlap_masks_for_mem_enc", 1),
    ("binarize_mask_from_pts", 1), ("multimask_output_for_tracking", 1),
    ("multimask_min_pt_num", 1), ("multimask_max_pt_num", 1),
    ("fixed_no_obj_ptr", 1), ("iou_prediction_use_sigmoid", 1),
    ("use_mask_input_as_output", 1), ("multimask_output_in_sam", 1),
    ("is_sam2_1", 1),
]

# .ggml dtype ints ARE GGML_TYPE enum values — verified by sam3_load_tensors()
# which does `static_cast<ggml_type>(dtype)` and feeds it to ggml_row_size().
DTYPE_TO_GGML = {
    0: GGML_TYPE_F32,
    1: GGML_TYPE_F16,
    2: GGML_TYPE_Q4_0,
    3: GGML_TYPE_Q4_1,
    8: GGML_TYPE_Q8_0,
}


def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"unexpected EOF (wanted {n} bytes, got {len(b)})")
    return b


def read_i32(f):
    return struct.unpack("<i", read_exact(f, 4))[0]


def read_u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_hparams(f, spec):
    hp = {}
    for name, count in spec:
        if count == 1:
            hp[name] = read_i32(f)
        else:
            hp[name] = [read_i32(f) for _ in range(count)]
    return hp


def read_tensor_meta(f):
    """Read one tensor record header. Returns (name, ne, ggml_type, nbytes, data_pos).

    The .ggml record stores dimensions already in ggml column-major order
    (converter wrote PyTorch shape reversed), so ne is the file shape verbatim:
    ne[0] is the innermost (row) dimension, exactly what ggml_new_tensor_*d
    expects and what the registered tensors use."""
    n_dims = read_i32(f)
    name_len = read_i32(f)
    dtype = read_i32(f)
    if dtype not in DTYPE_TO_GGML:
        raise ValueError(f"unsupported tensor dtype {dtype}")
    ne = [read_i32(f) for _ in range(n_dims)]   # ggml order: ne[0] innermost
    name = read_exact(f, name_len).decode("utf-8")
    # Record-header end is padded to 32-byte alignment; the tensor data starts
    # there. Keep the stream cursor at the header end — the data bytes are
    # fetched in the streaming pass via data_pos.
    pos = f.tell()
    pad = (ALIGN - pos % ALIGN) % ALIGN
    data_pos = pos + pad
    ttype = DTYPE_TO_GGML[dtype]
    return name, ne, ttype, ggml_nbytes(ttype, ne), data_pos


def read_tokenizer(f):
    """Read the embedded BPE tokenizer. Returns (vocab_by_id, merges)."""
    if read_u32(f) != TOK_MAGIC:
        raise ValueError("invalid tokenizer magic")
    n_vocab = read_i32(f)
    vocab = {}
    for _ in range(n_vocab):
        token_len = read_i32(f)
        token = read_exact(f, token_len).decode("utf-8")
        token_id = read_i32(f)
        vocab[token_id] = token
    n_merges = read_i32(f)
    merges = []
    for _ in range(n_merges):
        len_a = read_i32(f)
        a = read_exact(f, len_a).decode("utf-8")
        len_b = read_i32(f)
        b = read_exact(f, len_b).decode("utf-8")
        merges.append(f"{a} {b}")
    return vocab, merges


def convert_file(in_path, out_path):
    print(f"[1/2] scanning {in_path}")
    with open(in_path, "rb") as fin:
        magic = read_u32(fin)
        version = read_i32(fin)
        ftype = read_i32(fin)
        n_tensors = read_i32(fin)

        if magic == SAM3_MAGIC:
            arch = "sam3"
            hp = read_hparams(fin, SAM3_HPARAMS)
        elif magic == SAM2_MAGIC:
            hp = read_hparams(fin, SAM2_HPARAMS)
            if hp["backbone_type"] != 1:
                raise ValueError(f"unsupported SAM2 backbone_type={hp['backbone_type']}")
            arch = "sam2"
        else:
            raise ValueError(f"unknown magic 0x{magic:08x} (not a sam3/sam2 .ggml)")

        # First pass: collect tensor metadata (data bytes are streamed later).
        meta = []
        for i in range(n_tensors):
            name, ne, ttype, nbytes, data_pos = read_tensor_meta(fin)
            meta.append((name, ne, ttype, nbytes, data_pos))
            # Records are back-to-back: header (padded), then its data blob —
            # jump over the data to land on the next record header.
            fin.seek(data_pos + nbytes)
            if (i + 1) % 500 == 0:
                print(f"    scanned {i + 1}/{n_tensors} tensors")

        # Tokenizer (SAM3 only, and only when the text path is present).
        tok = None
        if arch == "sam3" and hp["visual_only"] == 0:
            tok = read_tokenizer(fin)

    w = GGUFWriter()
    w.add_str("general.architecture", arch)
    w.add_str("sam3.arch", arch)
    w.add_i32("sam3.version", version)
    w.add_i32("sam3.ftype", ftype)
    for name, val in hp.items():
        key = f"sam3.hparams.{name}"
        if isinstance(val, list):
            w.add_arr_i32(key, val)
        else:
            w.add_i32(key, val)
    if tok is not None:
        vocab, merges = tok
        w.add_arr_str("sam3.tokenizer.vocab", [vocab[i] for i in range(len(vocab))])
        w.add_arr_str("sam3.tokenizer.merges", merges)

    # Register tensor infos (data deferred to the streaming pass below).
    for name, ne, ttype, nbytes, data_pos in meta:
        w.add_tensor(name, ne, ttype)

    print(f"[2/2] writing {out_path} ({n_tensors} tensors)")
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        w.write_header_and_meta(fout)
        for i, (name, ne, ttype, nbytes, data_pos) in enumerate(meta):
            fin.seek(data_pos)
            data = read_exact(fin, nbytes)
            w.write_tensor_data(fout, i, data)
            if (i + 1) % 500 == 0:
                print(f"    wrote {i + 1}/{n_tensors} tensors")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"done: {arch} v{version} ftype={ftype}, {size_mb:.1f} MB -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="path to a .ggml model file")
    ap.add_argument("output", nargs="?", help="output .gguf path (default: same name)")
    ap.add_argument("--dir", metavar="DIR", help="repack every *.ggml in DIR")
    args = ap.parse_args()

    if args.dir:
        d = args.dir
        files = sorted(f for f in os.listdir(d) if f.endswith(".ggml"))
        if not files:
            print(f"no .ggml files in {d}")
            return
        for f in files:
            in_path = os.path.join(d, f)
            out_path = os.path.join(d, f[:-5] + ".gguf")
            try:
                convert_file(in_path, out_path)
            except Exception as e:
                print(f"FAILED {in_path}: {e}")
        return

    if not args.input:
        ap.error("either <input> or --dir is required")
    in_path = args.input
    out_path = args.output or in_path[:-5] + ".gguf"
    convert_file(in_path, out_path)


if __name__ == "__main__":
    main()
