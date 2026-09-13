#!/usr/bin/env python3
"""Differential corpus for the C++ tokenizer clean-pipeline check.

Uses the OFFICIAL sam3 tokenizer (tokenizer_ve.py SimpleTokenizer, clean="lower")
as the reference: basic_clean = ftfy.fix_text + html.unescape x2 + strip,
whitespace_clean, str.lower, CLIP BPE. Writes base64 corpus + expected token id
lists for tests/diff_tokenizer.cpp to compare against.

    python3 tests/diff_tokenizer_corpus.py /tmp/tok_diff
"""
import base64
import importlib.util
import os
import random
import sys
import types

# Load tokenizer_ve.py as a STANDALONE module (bypassing the sam3 package
# __init__, which pulls heavy deps like einops that the tokenizer never uses).
spec = importlib.util.spec_from_file_location(
    "tokenizer_ve_standalone",
    os.path.join(os.environ.get("SAM3_ROOT", "/home/asher/develop/code/dl/sam3"),
                 "sam3/model/tokenizer_ve.py"))
tokenizer_ve = importlib.util.module_from_spec(spec)

# tokenizer_ve.py imports iopath for g_pathmgr.open; stub it so the module can
# be loaded standalone (we pass a plain filesystem path).
iopath_stub = types.ModuleType("iopath")
common_stub = types.ModuleType("iopath.common")
file_io_stub = types.ModuleType("iopath.common.file_io")


class _PathMgr:
    def open(self, path, mode="r"):
        return open(path, mode)


file_io_stub.g_pathmgr = _PathMgr()
common_stub.file_io = file_io_stub
iopath_stub.common = common_stub
sys.modules.setdefault("iopath", iopath_stub)
sys.modules.setdefault("iopath.common", common_stub)
sys.modules.setdefault("iopath.common.file_io", file_io_stub)

spec.loader.exec_module(tokenizer_ve)
SimpleTokenizer = tokenizer_ve.SimpleTokenizer

SAM3_ROOT = os.environ.get("SAM3_ROOT", "/home/asher/develop/code/dl/sam3")


def build_corpus():
    c = [
        # plain / common
        "cat", "a red car", "The quick brown fox!", "don't stop", "123",
        "person, dog & cat", "  spaced   out  ", "", " ", "\t\n",
        # HTML entities
        "&amp;", "&lt;tag&gt;", "&amp;amp;", "&AMP", "&notit;", "&amp",
        "&#8212;", "&#x2019;", "&#0;", "&#13;", "&#x110000;", "&nbsp;",
        "a&nbsp;b", "&hellip; it&#x2019;s ﬂubberiﬁc!", "<b>bold</b>",
        "P&EACUTE;REZ", "&macr;\\_(ã\x83\x84)_/&macr;",
        # curly quotes / dashes
        "don’t “quote” me", "‘single’ — dash … ellipsis", "«guillemet»",
        # fullwidth / halfwidth
        "ＬＯＵＤ　ＮＯＩＳＥＳ", "Ｕﾀｰﾝ", "ｶﾀｶﾅ", "！？、", "（株）",
        # ligatures
        "ﬂuﬃeﬆ", "ﬁle", "ﬀ ﬃ ﬄ",
        # mojibake (latin-1 / sloppy-1252 families)
        "Ã©lÃ¨ve", "âœ” No problems", "Ã la mode", "voilÃ le travail",
        "Â£100", "sÃ³", "Ã©", "â€œquotedâ€", "fÃ cil", "Ã s",
        "cafÃ©", "Ã puszta", "Ã’ll", "ÅngstrÃ¶m", "Î©", "√Ç√∏√´",
        # C1 controls
        "\x91quoted\x92", "\x93dq\x94", "\x85 nel", "\x96 dash",
        # Unicode whitespace
        "a\u00a0b", "a\u3000b", "a\u2028b", "a\u2003b", "a\u000bb",
        # NFC / accents
        "cafe\u0301", "e\u0301\u0327x", "Ångström", "éàüöß",
        "가\u0301", "q̣̇",
        # Unicode lower / case
        "КАТ", "ΟΔΟΣ", "İstanbul", "ǅungla", "ⅠⅡⅢ", "ʼN",
        "ΣΊΣΥΦΟΣ", " indice\u0130 ",
        # misc unicode
        "📷 emoji", "обновление", "北京 北京", "में text", "ｱｲｳ ｴｵ",
        "mixed Ã©엌 двоичный", "\U0001F600", "ᵃᵇᶜ",
        # terminal escapes / control
        "\x1b[31mred\x1b[0m", "\x1b[36;44mblue\x1b[0m", "a\u0000b",
        "a\u0007b", "a\ufffeb", "a\u200bb", "a\ufeffb",
    ]
    rng = random.Random(1234)
    pool = ("cat dog red car ßé水ÀÂÃƒ„…†‡‰ŒœžŸ‚“”‘’š›™×÷¬"
            "😀γд έЅѕїј bolts 🌟\u0301\u0308\u0327 ¡¿«» ﬁﬂ "
            "&amp; &#x2019; ﾃｨ Â£ ™")
    for _ in range(120):
        n = rng.randint(1, 24)
        c.append("".join(rng.choice(pool) for _ in range(n)))
    return c


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tok_diff"
    os.makedirs(out_dir, exist_ok=True)

    tok = SimpleTokenizer(os.path.join(SAM3_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
                          context_length=32)
    corpus = build_corpus()
    expected = []
    for s in corpus:
        ids = [tok.sot_token_id] + tok.encode(s) + [tok.eot_token_id]
        if len(ids) > 32:
            ids = ids[:32]
            ids[-1] = tok.eot_token_id
        ids = ids + [0] * (32 - len(ids))
        expected.append(ids)

    with open(os.path.join(out_dir, "corpus.txt"), "w") as f:
        for s in corpus:
            f.write(base64.b64encode(s.encode("utf-8")).decode() + "\n")
    # cleaned expectations for DIFF_TEXT mode (official _clean_lower)
    with open(os.path.join(out_dir, "expected_clean.txt"), "w") as f:
        for s in corpus:
            f.write(base64.b64encode(tok.clean_fn(s).encode("utf-8")).decode() + "\n")
    with open(os.path.join(out_dir, "expected.txt"), "w") as f:
        for ids in expected:
            f.write(" ".join(map(str, ids)) + "\n")
    print("corpus: %d strings -> %s" % (len(corpus), out_dir))


if __name__ == "__main__":
    main()
