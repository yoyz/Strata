"""The GGUF pre-tokenizer names: llama.cpp gives DeepSeek-V3's regex sequence to more than one name.

`tokenizer.ggml.pre` is what tells the splitter which regex list to use.  DeepSeek-V4-Flash-0731 states
"joyai-llm"; llama.cpp's src/llama-vocab.cpp gives LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM, HUNYUAN_DENSE, JOYAI_LLM
and HY_V4 the same list.  Matching only names that start with "deepseek" silently used the Qwen pattern, which
splits digits and CJK differently and produces ids the model was not trained on.

Checked here without a model: the pieces the splitter makes, and (when the pack is on this computer) the ids of
three strings against llama.cpp's own llama-tokenize.

    python -m unittest tests.terminal.test_tokenizer_pre
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import strata_tokenizer as ST  # noqa: E402


def tiny(pre: str) -> ST.Tokenizer:
    """The real Tokenizer over DeepSeek's byte vocabulary and no merges: enough to see how text is split."""
    b2u = ST.bytes_to_unicode()
    return ST.Tokenizer([b2u[b] for b in range(256)], [], None, pre, {"eos": 1})


class PreNames(unittest.TestCase):
    def test_the_names_llama_cpp_gives_deepseek3_regex(self):
        self.assertEqual(ST.DEEPSEEK3_PRES, ("deepseek", "joyai", "hunyuan-dense", "hy_v4"))

    def test_joyai_llm_splits_as_deepseek3(self):
        # llama.cpp's DEEPSEEK3 sequence groups a run of 1..3 digits: 3.14 -> "3" ".", "14" (the Qwen pattern: "1", "4")
        ds, joy = tiny("deepseek3"), tiny("joyai-llm")
        for text in ("3.14", "12345", "don't 3.14", "東京 1234567890"):
            self.assertEqual(joy._split(text), ds._split(text), text)
        self.assertEqual(joy._split("3.14"), ["3", ".", "14"])

    def test_the_qwen_pattern_is_still_used_for_other_names(self):
        qwen = tiny("qwen35")
        self.assertEqual(qwen._split("3.14"), ["3", ".", "1", "4"])


class RealPack(unittest.TestCase):
    """When STRATA_DSV41_PACK points at a pack's tokenizer folder, its ids match llama.cpp's llama-tokenize."""

    def test_ids_match_llama_cpp(self):
        pack = os.environ.get("STRATA_DSV41_PACK")
        if not pack:
            self.skipTest("STRATA_DSV41_PACK is not set")
        tdir = Path(pack) / "tokenizer" if (Path(pack) / "tokenizer").is_dir() else Path(pack)
        vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tdir / "token_type.json").read_text())
        cfg = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
        tok = ST.Tokenizer(tokens, merges, types, cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)
        self.assertEqual(tok.pre, "joyai-llm")
        model = os.environ.get("STRATA_DSV41_MODEL")
        exe = os.environ.get("STRATA_LLAMA_TOKENIZE")
        if not (model and exe):
            self.skipTest("STRATA_DSV41_MODEL and STRATA_LLAMA_TOKENIZE are not set")
        for text in ("Hello, world! 12345", "don't 3.14 — naïve café 東京 1234567890",
                     "The capital of France is Paris."):
            got = subprocess.run([exe, "-m", model, "-p", text, "--ids", "--no-bos"],
                                 capture_output=True, text=True, check=True).stdout
            ids = json.loads(got.strip().splitlines()[-1])
            self.assertEqual(tok.encode(text, parse_special=False), ids, text)


if __name__ == "__main__":
    unittest.main()
