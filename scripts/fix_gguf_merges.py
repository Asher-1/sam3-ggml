#!/usr/bin/env python3
"""Patch the BPE merge table inside an existing SAM3 GGUF file (in place).

The original convert_sam3_to_ggml.py skipped every merges.txt line starting
with '#', silently dropping the 6 legitimate merges whose first part is '#'
("# #</w>", "# :</w>", ...).  Existing checkpoints therefore carry a
48888-entry merge table instead of the official 48894, and inputs like "##"
fail to merge into the single vocab token "##</w>" (id 15483) — caught by
tests/diff_tokenizer.cpp (211/212).

The converter is fixed now, but existing checkpoints cannot be regenerated
without the original sam3.pt, so this tool rewrites ONLY the
`sam3.tokenizer.merges` string-array KV, sourcing the merge rows from the
official bpe_simple_vocab_16e6.txt.gz exactly like the official
SimpleTokenizer does:

    merges = lines.split("\\n")[1 : 49152 - 256 - 2 + 1]

Everything else (KV order, tensor infos, tensor payload) is copied through
byte-for-byte; only the KV section length changes, so the data-segment
alignment padding is recomputed and the payload is streamed (bounded memory).

GGUF v3 layout assumption (written by ggml's gguf writer):
    header | KVs | tensor infos | align padding | data segment
Tensor-info offsets are relative to the data segment start, so they stay
valid as long as the data bytes are copied verbatim.

Usage:
    python3 scripts/fix_gguf_merges.py [--bpe-path PATH] [--dry-run] FILE...
"""
import argparse
import gzip
import os
import struct
import sys
import tempfile

# Fixed-size GGUF value types: type -> byte length
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_STR = 8
TYPE_ARR = 9
MERGES_KEY = "sam3.tokenizer.merges"
VOCAB_KEY = "sam3.tokenizer.vocab"

# Official-alignment hparams absent from checkpoints converted before the
# tracker-alignment work (select_closest_cond_frames cap + memory selection).
# Injected when missing; the C++ loaders fall back to SAM2-compatible
# defaults (-1 / 0) when the keys are absent.
HPARAM_KVS = [
    ("sam3.hparams.max_cond_frames_in_attn", 4),
    ("sam3.hparams.use_memory_selection", 1),
    ("sam3.hparams.mf_threshold_x100", 1),
]

DEFAULT_BPE = "/home/asher/develop/code/dl/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


def official_merges(bpe_path):
    """The exact merge list the official SimpleTokenizer builds from the
    official vocab file: split('\\n'), drop line 0, take 49152-256-2 rows."""
    with open(bpe_path, "rb") as fh:
        lines = gzip.open(fh).read().decode("utf-8").split("\n")
    merges = lines[1 : 49152 - 256 - 2 + 1]
    out = []
    for m in merges:
        parts = m.split()  # official: tuple(merge.split())
        if len(parts) != 2:
            raise SystemExit(f"unexpected merge row: {m!r}")
        out.append((parts[0], parts[1]))
    return out


class Reader:
    def __init__(self, f):
        self.f = f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def s(self):
        n = self.u64()
        return self.f.read(n)  # raw bytes (merge rows are ASCII-safe)


def encode_str(b):
    return struct.pack("<Q", len(b)) + b


def encode_kv_merges(merges):
    """Complete KV: key string + u32 type(9 array) + u32 elem type(8 string)
    + u64 count + count * string."""
    out = [encode_str(MERGES_KEY.encode("utf-8")),
           struct.pack("<IIQ", TYPE_ARR, TYPE_STR, len(merges))]
    for a, b in merges:
        out.append(encode_str((a + " " + b).encode("utf-8")))
    return b"".join(out)


def encode_kv_i32(key, value):
    """Complete KV: key string + u32 type(5 = i32) + i32 payload."""
    return encode_str(key.encode("utf-8")) + struct.pack("<Ii", 5, value)


def scan_kv(reader, expect_key):
    """Read one KV. Returns (key_bytes, value_bytes, key, parsed_type,
    str_items_or_None)."""
    start = reader.f.tell()
    key = reader.s()
    vtype = reader.u32()
    if vtype == TYPE_STR:
        val = reader.s()
        end = reader.f.tell()
        return key, val, key.decode("utf-8", "replace"), vtype, None, (start, end)
    if vtype == TYPE_ARR:
        et = reader.u32()
        n = reader.u64()
        items = None
        if et == TYPE_STR:
            items = [reader.s() for _ in range(n)]
        elif et in FIXED:
            reader.f.read(n * FIXED[et])
        else:
            raise SystemExit(f"unsupported array elem type {et} in KV {key!r}")
        end = reader.f.tell()
        return key, None, key.decode("utf-8", "replace"), vtype, items, (start, end)
    if vtype in FIXED:
        val = reader.f.read(FIXED[vtype])
        end = reader.f.tell()
        return key, val, key.decode("utf-8", "replace"), vtype, None, (start, end)
    raise SystemExit(f"unsupported KV type {vtype} for {key!r}")


def align_up(x, a):
    return (x + a - 1) // a * a


def patch_file(path, merges, dry_run):
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        r = Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version = r.u32()
        if version != 3:
            raise SystemExit(f"{path}: unsupported GGUF version {version}")
        n_tensors = r.u64()
        n_kv = r.u64()
        header_pos = f.tell()  # start of KV section

        kvs = []
        merges_items = None
        merges_range = None
        alignment = 32
        for _ in range(n_kv):
            key, val, key_s, vtype, items, rng = scan_kv(r, MERGES_KEY)
            if key_s == "general.alignment" and vtype == 4:
                alignment = struct.unpack("<I", val)[0]
            if key_s == MERGES_KEY:
                if items is None:
                    raise SystemExit(f"{path}: {MERGES_KEY} is not a string array")
                merges_items = items
                merges_range = rng
            kvs.append((key, val, key_s, vtype, items, rng))

        kv_end = f.tell()
        # Walk the tensor infos to find their true end (the alignment padding
        # after them must NOT be copied — the data segment starts at
        # align_up(infos_end) and the padding size changes with the new KV
        # length).
        for _ in range(n_tensors):
            r.s()          # name
            n_dims = r.u32()
            f.read(8 * n_dims)  # shape (u64 each)
            r.u32()        # type
            r.u64()        # data offset (relative to data segment start)
        infos_end = f.tell()
        data_start = align_up(infos_end, alignment)
        if data_start > fsize or data_start < kv_end:
            raise SystemExit(f"{path}: corrupt file (data start {data_start} "
                             f"beyond EOF {fsize})")

        if merges_items is None:
            raise SystemExit(f"{path}: {MERGES_KEY} not found (visual-only model?)")

        old_rows = []
        for raw in merges_items:
            s = raw.decode("utf-8")
            p = s.split(" ", 1)
            if len(p) != 2:
                raise SystemExit(f"{path}: malformed merge row {s!r}")
            old_rows.append((p[0], p[1]))
        # Existing rows must be either already-patched (official table) or the
        # official table minus the '#'-first rows (the pre-fix converter
        # output). Anything else means an unexpected file — abort.
        expect = [m for m in merges if not m[0].startswith("#")]
        already_patched = old_rows == merges
        if not already_patched and old_rows != expect:
            raise SystemExit(
                f"{path}: existing merge table does not match official-minus-"
                f"'#' rows ({len(old_rows)} vs {len(expect)}); refusing to patch"
            )
        # Missing official-alignment hparams (idempotent). present_keys must
        # use the decoded strings (kvs[0] is the raw bytes key).
        present_keys = {key_s for _, _, key_s, _, _, _ in kvs}
        missing_kvs = [(k, v) for k, v in HPARAM_KVS if k not in present_keys]
        extra_kvs = b"".join(encode_kv_i32(k, v) for k, v in missing_kvs)
        if already_patched and not extra_kvs:
            print(f"{path}: already patched (merges {len(old_rows)}, "
                  f"hparams present); nothing to do")
            return
        if already_patched:
            print(f"{path}: merges already official ({len(old_rows)}); "
                  f"adding {len(missing_kvs)} alignment hparams")
        else:
            to_add = [m for m in merges if m[0].startswith("#")]
            print(f"{path}: merges {len(old_rows)} -> {len(merges)} "
                  f"(adding {len(to_add)}: {[' '.join(m) for m in to_add]})")
        if dry_run:
            return

        # Build the new file: header + KVs (merges replaced) + tensor infos
        # (clean, no padding) + recomputed alignment padding + verbatim data
        # segment (its start moves, but the relative tensor offsets do not).
        new_kv_bytes = bytearray()
        for key, val, key_s, vtype, items, rng in kvs:
            if key_s == MERGES_KEY:
                continue
            f.seek(rng[0])
            new_kv_bytes += f.read(rng[1] - rng[0])
        new_kv_bytes += encode_kv_merges(merges)
        new_kv_bytes += extra_kvs

        infos_size = infos_end - kv_end
        new_data_start = align_up(header_pos + len(new_kv_bytes) + infos_size,
                                  alignment)

        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                        suffix=".patched")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(magic)
                out.write(struct.pack("<IQQ", version, n_tensors,
                                      n_kv + len(missing_kvs)))
                out.write(new_kv_bytes)
                # tensor infos: clean raw bytes (no original padding)
                f.seek(kv_end)
                remaining = infos_size
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        raise SystemExit(f"{path}: unexpected EOF in tensor infos")
                    out.write(chunk)
                    remaining -= len(chunk)
                cur = out.tell()
                assert cur <= new_data_start
                out.write(b"\0" * (new_data_start - cur))
                # data segment: verbatim, streamed
                f.seek(data_start)
                left = fsize - data_start
                while left > 0:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
                    left -= len(chunk)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        print(f"{path}: patched OK ({fsize} -> {os.path.getsize(path)} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="GGUF files to patch (in place)")
    ap.add_argument("--bpe-path", default=DEFAULT_BPE,
                    help="official bpe_simple_vocab_16e6.txt.gz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    merges = official_merges(args.bpe_path)
    hash_first = sum(1 for m in merges if m[0].startswith("#"))
    print(f"official merges: {len(merges)} rows, {hash_first} start with '#'")
    for path in args.files:
        patch_file(path, merges, args.dry_run)


if __name__ == "__main__":
    main()
