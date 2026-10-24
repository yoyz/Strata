# Porting the DeepSeek engine to DeepSeek-V4-Flash-0731 (`deepseek4`)

Status: **experimental, not usable.** The packer accepts the model, the engine loads it, decodes, and runs the
model's own sparse attention (the compressor and the indexer) — but a real pack is still missing (item 3), and with it
any comparison against llama.cpp. This note is the restart point: what is done, what is left, and how to build and
run it.

## Where this stands

Branch `f-dsv41-ds4`, worktree `/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4`:

| commit | what |
|---|---|
| `335fcfa1` | the packer accepts `deepseek4`; the engine loads and decodes a pack (test rewrites, below) |
| `f66aafa4` | the head hyper-connection and the hash layers' routing |
| `990bb791` | this note and the scripted test pack |
| `03e7a42a` | the compressor: two halves a row (CSA's overlap), the position table, and the ring that keeps a group across a chunk |
| `this commit` | the indexer's own compressor (the LID keys), its rotation, and every compressed row on the ratio-128 layers |

Measured 2026-10-24, RX 9070 XT (16 GiB, gfx1201), 125 GiB RAM, one NVMe, the Qwen server stopped, the test pack
(no expert profile, cold store): dense weights 6.7 s, whole load 12.6 s, 763 of 11,008 experts in VRAM (6.1 GB), the
rest read in place from the files; a 3-token prompt in 2.24 s, then 2.27 tokens/s decoding, 1.33 GB read from the
disk a token. With the indexer in, a 2200-token prompt reads at 199 tokens/s on a cold store (319 warm) and
decodes at 16-18 tokens/s, with 550 compressed rows and the top-512 selection over them. Output is still wrong: the test pack's two substituted
layers (item 3) put wrong values in two of the 43.

The compressor is checked against a double reference written from llama.cpp's two compress graphs
(`tests/dsv41/compress_test.cpp`, 7 runs: CSA with the table, without it, a token a call, several groups a call, V4.1's
ratio 4, HCA ratio 128 and HCA across chunk boundaries): every group matches to 2e-5. Matching llama.cpp's own numbers
end to end needs a real pack first (item 3), because 2 of the 43 layers still carry the test pack's substitution.

Everything below marked *static* is read from the code, not measured on a card.

## Restart: build and run

```sh
# build (the ROCm SDK the Qwen build uses, gfx1201)
cd /mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4
V=/mnt/nvme_lexar_0/ollama/build/Strata-0.1.40.3/.venv/lib/python3.14/site-packages
cmake -G Ninja -S . -B /mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41 \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_HIP_COMPILER=$V/_rocm_sdk_devel/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=$V/_rocm_sdk_devel \
  -DCMAKE_HIP_FLAGS="--rocm-path=$V/_rocm_sdk_devel --rocm-device-lib-path=$V/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode" \
  -DCMAKE_PREFIX_PATH="$V/_rocm_sdk_devel;$V/_rocm_sdk_libraries_gfx120X_all" \
  -DSTRATA_GGML_DIR=/mnt/nvme_lexar_0/ollama/build/Strata-0.1.40.3/third_party/llama.cpp \
  -DSTRATA_NATIVE_EXPERTS=ON
HIP_PLATFORM=amd ninja -j 12 -C /mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41 strata-dsv41

# the compressors against their double reference (one GPU, no model)
HIP_PLATFORM=amd ninja -j 12 -C /mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41 ds4_compress_test
/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41/ds4_compress_test

# the pack (3 MB of metadata; the 91 GB of weights stay where they are)
python3 tools/dsv41/pack_gguf.py \
  --model /home/ollama/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ2_XXS \
  --out /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack
python3 tools/dsv41/ds4_testpack.py /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack \
  /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack-test      # TEST ONLY, see below

# run (the GPU must be free: stop the Qwen server first)
export LD_LIBRARY_PATH=$V/_rocm_sdk_devel/lib:$V/_rocm_sdk_devel/lib/rocm_sysdeps/lib:$V/_rocm_sdk_libraries_gfx120X_all/lib
/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41/strata-dsv41 \
  --pack /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack-test \
  --ids 100,200,300 --n 4 --max-context 8192 --reserve-mib 1024 \
  --expert-profile none --dspark none --cpu-threads 24
```

`--pack-info` prints the pack without touching a card. `--dump-logits FILE` and `--score FILE --score-out FILE` are
the comparison hooks for the validation step.

## To do

- [x] **1. The compressor: CSA (ratio 4) and HCA (ratio 128), with the position table.** Done in `03e7a42a`.
  - `src/dsv41/ds4_kernels.cu`: `compress_kernel` / `compress_win` take `coff` and `slots`, write a token's whole
    `coff · head` row (the position table added to the score as the row is written), pool a group over the rows it
    reads - the previous group's first halves and its own second halves (CSA), or its own `ratio` rows (HCA, V4.1) -
    per value, `acc / den`, then `rsqrt(mean + eps) · w`.
  - `src/dsv41/ds4_model.cu`: `Layer::c_ape`, `c_coff`, `c_slots`, `c_wide`, `c_n()`; the pack's `attn.compressor.wkv`
    row count decides `coff` (V4.1 512 → 1, deepseek4's ratio-4 1024 → 2; anything else is an error), the ring is
    `coff · ratio` rows (2·ratio keeps the previous group until it has been read), the APE is loaded next to the
    compressor's matrices, and every size that used `ratio · hd` (`reset`, the checkpoints, the slots, the speculative
    window's snapshot of the state and its replay of the units' raw rows) uses `c_n()` / `c_wide`.
  - Checked: `ds4_compress_test` (below). Not yet checked against llama.cpp: that needs a real pack (item 3), because
    the test pack's two substituted layers sit in the middle of the 43 and everything after them is wrong.
  - The doubt about `2·head` rows in `Src`: none needed - the pooled row is `head` wide and the source cache's rows
    are unchanged; only the compressor's own state got wider.

- [x] **2. The indexer's own compressor (LID), 21 ratio-4 layers.** Done in this commit.
  - `Layer::i_comp`, `i_cwkv`/`i_cwgate`, `i_cnorm`/`i_ape`, `i_kv`/`i_score` (the same shape rules as the KV
    compressor: two halves of 128, a ring of `2·ratio` rows); the keys are its pooled rows, normed and roped with the
    compressor's own weights, packed to the index cache as before. `indexer.wk` is not read when the pack has this.
  - `hadamard_rows` (ds4_kernels.cu): the orthonormal Walsh-Hadamard llama.cpp applies to the indexer's queries and
    keys (`ggml_gen_hadamard`). It is part of the model, not of llama.cpp's cache quantisation: `attn_rot_k` is set
    for `LLM_ARCH_DEEPSEEK4` unconditionally (`llama-kv-cache.cpp`), and DeepSeek's own indexer computes q/k on a
    quantised grid. Both sides are rotated, so the scores are unchanged and only the FP4 rounding differs.
  - `index_score` now takes 64 heads (32 at a time); the old limit of 32 refused the model's indexer.
  - The ratio-128 layers have no indexer: they attend to every compressed row (llama.cpp's `build_hca_attention`).
    That needed a path of its own (`i_comp_model`): before, a `ratio > 0` layer with no index source took the count
    the last indexer left and *its* row ids, which is wrong for a cache of 128-token rows. Capped at the selection
    buffer (512 rows), which is exact to a 65,536-token context.
  - `pack_gguf.py` writes `index_source_layer_ids` = the 21 layers that carry the compressor (V4.1's `indexer.attn_k`
    when a pack has that instead).
  - The indexer's state is part of the checkpoints, the conversation slots (a pack without it writes the same bytes
    as before), the mid-run `feed_mark` and the speculative window's replay.
  - Not checked against llama.cpp: the top-512 selection only starts to matter past 512 compressed rows (about 2,048
    tokens), and the test pack's two substituted layers are in the way before that. What is measured: a 2200-token
    prompt runs the selection (550 rows -> 512) without a fault, at 199 tokens/s.

- [ ] **3. MXFP4 mixed with the I-quants (the real pack has it).**
  - Layers 26 and 42 of UD-IQ2_XXS have MXFP4 down experts; `expert_layout_mx` rejects MXFP4 beside IQ gate/up.
  - Either read each role with its own kernels (the `ds4_gguf.cu` path branches on `mx_experts(lay)`, all three
    roles MXFP4, so this means a per-role dispatch in the CPU lane and the GPU windows), or requantise those two
    down tensors in the packer.
  - Done when `ds4_testpack.py`'s down substitution is not needed.

- [ ] **4. Validation against llama.cpp.**
  - A fixed token sequence, `--dump-logits` from the engine, llama.cpp's logits for the same ids (the pinned
    checkout has `models/deepseek4.cpp`; build `llama-cli`/`llama-perplexity` from it or use the binary the GGUF was
    tested with).
  - Compare the first tokens' logits layer by layer if possible (`--dump-logits` is final only; a per-layer tap may
    need a debug hook).
  - Then 8K context, then speed with an expert profile (`tools/dsv41` has the profile tools; the shipped
    `profile-2.00.bin` is for the IST-DAS sets, not this model).

- [ ] **5. Not started, after the engine matches:** the model in `setup.py`/`strata_control` (family, pin, sizes),
  the server hand-off, and a `strata-dsv41` package for the feed.

## Test-only rewrite in `ds4_testpack.py` (remove when item 3 lands)

1. layers 26 and 42's down experts point at layer 25's / 41's (same shape, wrong values): the engine cannot mix
   MXFP4 with the I-quants.

It is in `tensors.txt` only; the weights are not touched. A real pack must never be built this way. (The compressor
matrices are read whole since item 1: `ds4_testpack.py` no longer cuts their first half away.)

## The model

From the GGUF's own metadata (`UD-IQ2_XXS`, 3 shards, 90.9 GB of weights + a 10.9 GB DSpark drafter):

| | |
|---|---|
| architecture | `deepseek4` (llama.cpp's id) |
| layers | 43 |
| experts | 256 routed, 6 a token, 1 shared; `hash_layer_count` 3 |
| hidden / heads | 4096 / 64, head 512, rope 64, q_lora 1024, o_lora 1024 × 8 groups |
| hyper-connections | 4 (`hc_*` per layer; the engine already reads them) |
| compress ratios | `0,0,4,128,4,128,…,4` (43 entries), then 3 more for the drafter |
| indexer | 64 heads × 128, top-k 512, on the 21 ratio-4 layers |
| context | 1,048,576 (the engine is checked at 8192, as V4.1 is) |
| quant | UD-IQ2_XXS: IQ2_XXS/IQ3_XXS experts, layer 26 IQ2_S + MXFP4, Q4–Q6_K dense |

The engine implements DeepSeek-V4.1-Flash (`deepseek41`: 40 layers, 384 experts, an Engram n-gram table). The packer
translates the `deepseek4.*` keys to the `ds4.txt` the engine reads; the geometry (layers, experts, ratios, heads)
arrives correctly — that is why the model loads.

## The four differences, in full

### 1. Compressor: per layer, with the learned position table, and CSA's overlap — **done**

V4.1: a few source layers own the compressed KV of a ratio; the following layers read it. The engine's
`kv_source_layer_ids` model.

V4-Flash: every layer with `ratio > 0` has its own `attn_compressor_kv/gate/norm/ape`. The packer writes
`kv_source_layer_ids = [2..42]`.

- **ratio 128 (HCA)**: `wkv`/`wgate` `[4096, 512]`, `ape` `[512, 128]`. A block of 128 tokens: softmax over its 128
  scores (each `wgate(x) + ape[slot]`), weighted sum of the 128 `head`-wide values, RMS norm, RoPE.
  llama.cpp: `build_hca_compressed_kv_from_state`.
- **ratio 4 (CSA)**: `wkv`/`wgate` `[4096, 1024]`, `ape` `[1024, 4]`. A block of four tokens mixes eight values:
  the previous block's four tokens' first half and its own four tokens' second half, with the eight scores
  (softmax), sum to one `head` vector, RMS norm, RoPE. llama.cpp: `build_overlap_compressed_kv_from_state`, and the
  index plan in `dsv4_state_read_idxs` (`src/llama-kv-cache-dsv4.cpp`).

The softmax is per value, over the rows a group reads - the engine's original `compress_win` already had that shape;
what it lacked was the position table, a row of `2·head` on a ratio-4 layer (the first `head` values being what the
next group reads as its previous half, the last `head` what its own group reads), and the `2·ratio` ring rows that
keep the previous group until it has been read. `coff` comes from the pack (`attn.compressor.wkv`'s row count over
`head`), so a V4.1 pack still runs the old shapes.

### 2. Indexer: it compresses its own keys — **done**

V4.1: `indexer.wk` projects the main compressed rows into index keys.

V4-Flash: 21 ratio-4 layers carry `indexer_compressor_{ape,kv,gate,norm}` — the same CSA algorithm at 128 wide —
and those compressed rows are the keys; `indexer.attn_k` does not exist. The candidate-block step
(`cand_src`/`uses_cand`) does not exist either (the engine's `uses_cand` stays false).

The engine runs the compressor per layer (`Layer::i_comp`), rotates both sides of the indexer's dot products with the
Walsh-Hadamard the model uses (`hadamard_rows`), and lets the ratio-128 layers see every compressed row, since they
have no indexer at all (llama.cpp's `build_hca_attention`; its `build_csa_lid_attention` is the ratio-4 path, and
`attn_rot_k` is always on for this architecture, so the rotation is the model's, not a cache-quantisation detail).

### 3. The first three layers route by token id — **done**

`ffn_gate_tid2eid` (I32 `[6, 129280]`) on layers 0–2: the experts are `table[token]`, the weights the gate's scores
at them, no router bias. The packer carries it (`layers.N.ffn.gate.tid2eid`), the engine uploads the window's ids
(`Dev::tids`) and `route(..., tids, tid2eid)` takes them. `ffn.gate.bias` is zero on these layers (the engine now
makes a zero bias where the model has none).

### 4. The head's hyper-connection — **done**

`output_hc_fn` `[4·dim, 4]`, `base` `[4]`, `scale` `[1]`: flatten the four states, RMS norm, one linear to four
values, `sigmoid(scale·mix + base) + eps`, weighted sum, then `output_norm` and the head. `head_pre()` + the
`head_hc_pre` kernel, at all three head paths.

## Packer changes (done)

`tools/dsv41/pack_gguf.py`:

- accepts `deepseek4` (`MODEL_ARCHS`);
- maps `attn.compressor.ape`, `attn.indexer.compressor.{ape,wkv,wgate,norm}`, `head.hc_*`;
- carries `ffn_gate_tid2eid` as I32 (`layers.N.ffn.gate.tid2eid`) and `hash_layer_count`;
- writes a header-only `engram.bin` when the model has no Engram (the engine opens it unconditionally);
- takes the tokenizer `pre` from the GGUF (`joyai-llm`) instead of assuming `deepseek3`;
- writes `kv_source_layer_ids` for every ratio>0 layer and `index_source_layer_ids` for the 21 that carry the
  indexer's compressor.

## What the engine does with the pack today

`strata-dsv41` reads `ds4.txt` + `tensors.txt`, places the dense weights and as many experts as VRAM holds, reads
the rest in place from the GGUF, and runs the model's own graph: the window, the compressed rows (each layer its own,
from the two halves of its group and the position table), the ratio-4 layers' top-512 selection from the indexer's
own compressed keys, every compressed row on the ratio-128 layers, hyper-connections per layer, MoE with top-k or the
hash table, and the head with its hyper-connection. What is left is the real pack (item 3: layers 26 and 42 still
carry the test pack's substituted down experts) and the comparison against llama.cpp (item 4). The compressors are
checked against a double reference (`ds4_compress_test`); the indexer's selection is not checked yet.

## Cautions

- The engine package for the feed (`strata-dsv41-…tar.gz`) does not exist for this model; this is a source build.
- `strata launch`/`setup.py` do not know the model yet (item 5). They still serve V4.1 only.
- The test rewrites change values, not shapes: never ship such a pack.
