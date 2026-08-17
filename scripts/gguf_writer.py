#!/usr/bin/env python3
"""Shared GGUF v3 writer for the sam3.cpp conversion pipeline.

Implements the GGUF v3 container format (magic "GGUF", version 3) as
specified by ggml/src/gguf.cpp, so that every model file produced by this
repo loads through the same `gguf_init_from_file` path as the other
ggml-based projects (face-detect-ggml, free-splatter.cpp, OpenPCDet-GGML)
— no custom binary formats anywhere in the ecosystem.

File layout (all little-endian):
  header:  "GGUF" (4B) | u32 version | u64 n_tensors | u64 n_kv
  KV:      per key: string key | u32 type | value
  tensors: per tensor: string name | u32 n_dims | u64 dims[n_dims]
                       | u32 ggml_type | u64 offset
  data:    pad to alignment (default 32), then tensor blobs in offset order,
           each padded to alignment.

GGUF types (gguf.h enum gguf_type):
  UINT8=0 INT8=1 UINT16=2 INT16=3 UINT32=4 INT32=5 FLOAT32=6 BOOL=7
  STRING=8 ARRAY=9 UINT64=10 INT64=11 FLOAT64=12

ggml tensor types used by this project (ggml.h enum ggml_type):
  F32=0 F16=1 Q4_0=2 Q4_1=3 Q8_0=8
"""

import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# gguf_type values
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# ggml_type values used by this project
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8

# (type_size, block_size) per ggml_type — matches ggml_type_size()/ggml_blck_size()
_GGML_TYPE_BLOCK = {
    GGML_TYPE_F32:   (4, 1),
    GGML_TYPE_F16:   (2, 1),
    GGML_TYPE_Q4_0:  (18, 32),   # 2B scale + 16B quants per 32 elements
    GGML_TYPE_Q4_1:  (20, 32),   # 2x2B scales + 16B quants per 32 elements
    GGML_TYPE_Q8_0:  (34, 32),   # 2B scale + 32B quants per 32 elements
}


def ggml_nbytes(ggml_type: int, ne) -> int:
    """Total byte size of a contiguous ggml tensor of shape ne (ne[0] innermost).

    Mirrors ggml_nbytes(): row_size(type, ne[0]) * prod(ne[1:]).
    """
    if ggml_type not in _GGML_TYPE_BLOCK:
        raise ValueError(f"unsupported ggml_type {ggml_type}")
    type_size, blck_size = _GGML_TYPE_BLOCK[ggml_type]
    assert ne[0] % blck_size == 0, f"ne[0]={ne[0]} not divisible by block size {blck_size}"
    nbytes = type_size * (ne[0] // blck_size)
    for d in ne[1:]:
        nbytes *= d
    return nbytes


class GGUFWriter:
    """Streams KV pairs and tensors, then writes a complete GGUF v3 file.

    Tensor data is supplied as bytes (already in ggml layout); the writer
    computes the offset table and 32-byte alignment, identical to what
    gguf_add_tensor()/gguf_write_to_file() would produce.
    """

    def __init__(self, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self._kv = []            # (key, type, value_bytes)
        self._tensors = []       # (name, dims, ggml_type)
        self._tensor_data = []   # optional per-tensor data bytes

    @staticmethod
    def _pad(n: int, alignment: int) -> int:
        return (n + alignment - 1) & ~(alignment - 1)

    # ── KV helpers ───────────────────────────────────────────────────────

    def _add_raw(self, key: str, gguf_type: int, value_bytes: bytes):
        assert key
        self._kv.append((key, gguf_type, value_bytes))

    @staticmethod
    def _enc_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def add_u32(self, key: str, val: int):
        self._add_raw(key, GGUF_TYPE_UINT32, struct.pack("<I", val))

    def add_i32(self, key: str, val: int):
        self._add_raw(key, GGUF_TYPE_INT32, struct.pack("<i", val))

    def add_f32(self, key: str, val: float):
        self._add_raw(key, GGUF_TYPE_FLOAT32, struct.pack("<f", val))

    def add_str(self, key: str, val: str):
        self._add_raw(key, GGUF_TYPE_STRING, self._enc_str(val))

    def add_arr_i32(self, key: str, vals):
        payload = b"".join(struct.pack("<i", v) for v in vals)
        self._add_raw(key, GGUF_TYPE_ARRAY,
                      struct.pack("<IQ", GGUF_TYPE_INT32, len(vals)) + payload)

    def add_arr_f32(self, key: str, vals):
        payload = b"".join(struct.pack("<f", v) for v in vals)
        self._add_raw(key, GGUF_TYPE_ARRAY,
                      struct.pack("<IQ", GGUF_TYPE_FLOAT32, len(vals)) + payload)

    def add_arr_str(self, key: str, vals):
        payload = b"".join(self._enc_str(v) for v in vals)
        self._add_raw(key, GGUF_TYPE_ARRAY,
                      struct.pack("<IQ", GGUF_TYPE_STRING, len(vals)) + payload)

    # ── Tensor helper ────────────────────────────────────────────────────

    def add_tensor(self, name: str, ne, ggml_type: int, data: bytes = None):
        """Register a tensor. ne is the ggml shape (ne[0] = innermost dim).
        data may be deferred (None) for streaming pipelines: pass it later to
        write_tensor_data()."""
        assert len(ne) >= 1 and len(ne) <= 4
        nbytes = ggml_nbytes(ggml_type, ne)
        if data is not None:
            assert len(data) == nbytes, (
                f"tensor '{name}': nbytes {nbytes} != data {len(data)}"
            )
        self._tensors.append((name, list(ne), ggml_type))
        self._tensor_data.append(data)

    # ── Serialization (streaming: never holds tensor data in memory) ────

    def _kv_bytes(self, key: str, gguf_type: int, value: bytes) -> bytes:
        return self._enc_str(key) + struct.pack("<I", gguf_type) + value

    def write_header_and_meta(self, f):
        """Write magic + version + counts + KV pairs + tensor infos + alignment
        padding. After this call the caller must write each tensor's data via
        write_tensor_data(), in registration order."""
        n_kv = len(self._kv)
        n_tensors = len(self._tensors)

        # header
        out = bytearray()
        out += GGUF_MAGIC
        out += struct.pack("<I", GGUF_VERSION)
        out += struct.pack("<Q", n_tensors)
        out += struct.pack("<Q", n_kv)

        # KV pairs
        for key, t, val in self._kv:
            out += self._kv_bytes(key, t, val)

        # tensor info (offset is relative to the start of the data section)
        offset = 0
        infos = []
        for name, ne, t in self._tensors:
            nbytes = ggml_nbytes(t, ne)
            infos.append((name, ne, t, offset))
            offset += self._pad(nbytes, self.alignment)
        for name, ne, t, off in infos:
            out += self._enc_str(name)
            out += struct.pack("<I", len(ne))
            for d in ne:
                out += struct.pack("<Q", d)
            out += struct.pack("<I", t)
            out += struct.pack("<Q", off)

        # data section (aligned)
        pad = (self.alignment - len(out) % self.alignment) % self.alignment
        out += b"\x00" * pad
        f.write(out)

    def write_tensor_data(self, f, index: int, data: bytes = None):
        """Write tensor[index]'s data blob followed by alignment padding.
        data may be supplied here if it was deferred in add_tensor()."""
        name, ne, t = self._tensors[index]
        nbytes = ggml_nbytes(t, ne)
        d = data if data is not None else self._tensor_data[index]
        assert d is not None and len(d) == nbytes, \
            f"tensor '{name}': expected {nbytes} bytes, got {None if d is None else len(d)}"
        f.write(d)
        pad = self._pad(nbytes, self.alignment) - nbytes
        if pad:
            f.write(b"\x00" * pad)

    def write(self, path: str):
        with open(path, "wb") as f:
            self.write_header_and_meta(f)
            for i in range(len(self._tensors)):
                self.write_tensor_data(f, i)
