# Porting the DeepSeek engine to DeepSeek-V4-Flash-0731 (`deepseek4`)

Status: **it answers, and it matches llama.cpp to 2-5% of a vector's L2.** A real pack builds and runs, and the
model answers the smoke question ("The capital of France is Paris.") and writes coherent text.  Comparing every stage
of a layer against llama.cpp's eval-callback found four bugs (below); what is left over is the engine's FP8 rounding of
the window rows and of the attention output, which DeepSeek's own inference does and llama.cpp does not.  This note is
the restart point: what is done, what is left, and how to build, run and compare it.

## Where this stands

Branch `f-dsv41-ds4`, worktree `/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4`:

| commit | what |
|---|---|
| `335fcfa1` | the packer accepts `deepseek4`; the engine loads and decodes a pack (test rewrites, below) |
| `f66aafa4` | the head hyper-connection and the hash layers' routing |
| `990bb791` | this note and the scripted test pack |
| `03e7a42a` | the compressor: two halves a row (CSA's overlap), the position table, and the ring that keeps a group across a chunk |
| `a751ca6e` | the indexer's own compressor (the LID keys), its rotation, and every compressed row on the ratio-128 layers |
| `649612e2` | item 3 (MXFP4 beside i-quant gate/up) and the hash layers' routing on the async window path |
| `332c09c2` | item 4: the per-head q norm, the hyper-connection pre-mix, the CPU lane's i-quant guard, the chunk path's token ids - it answers |

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

# the pack (3 MB of metadata; the 91 GB of weights stay where they are).  Pack 3 is the first one built by a packer that
# writes hc_lag 0 - a pack made before 332c09c2 gives the old pre-mix and wrong output; re-pack it.
python3 tools/dsv41/pack_gguf.py \
  --model /mnt/nvme_lexar_0/ollama_user_cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ2_XXS \
  --out /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack

# run (the GPU must be free: stop the Qwen server first)
export LD_LIBRARY_PATH=$V/_rocm_sdk_devel/lib:$V/_rocm_sdk_devel/lib/rocm_sysdeps/lib:$V/_rocm_sdk_libraries_gfx120X_all/lib
/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41/strata-dsv41 \
  --pack /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack3 \
  --ids 23166 --n 4 --max-context 8192 --reserve-mib 1024 \
  --expert-profile none --dspark none --cpu-threads 24

# the smoke question, through the pack's own tokenizer (the model must answer with Paris)
python3 tools/dsv41/smoke.py --exe /mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41/strata-dsv41 \
  --pack /mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack3 --tokens 16
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

- [x] **3. MXFP4 mixed with the I-quants (the real pack has it).** Done in this commit.
  - From the GGUF: layers 26 and 42 of UD-IQ2_XXS are **IQ2_S (26) / IQ2_XXS (42) gate and up with an MXFP4 down**;
    every other layer has one type for all three roles (IQ3_XXS down beside IQ2_XXS gate/up is common and already
    works - `lay.K[r]` is per role).
  - Blocked in two places: `expert_layout_mx` refuses MXFP4 beside the i-quants ("MXFP4 does not mix with the
    others"), and every dispatch asks `gg::mx_experts(lay)`, which means all three roles MXFP4 (`gg::expert_type`
    already answers per role).
  - Device kernels: both families take the same activations - `q8_quant`'s `xq`/`xd` for gate/up and `hq`/`hd` for
    the down - and read `lay.K[r]`, so the work is one dispatch per stage instead of one per layer:
    `moe_gate_up`, `moe_down`, `win_experts` (gate/up and down in one call), `set_gate_up` and `set_down`.
  - CPU lane: the two lanes differ in their activation quantizer (`gg::mx_quant` vs `gg::iq_cpu_quant`) and row
    stride, so split `cpu_experts_mx` / `cpu_experts_iq` into a gate/up half (x -> h, the swiglu) and a down half
    (h -> y) and run one of each for a mixed layer; an entry's token batch becomes
    `min(gg::kMxCpuMaxN, kEntMax)` (see `ent_cap`).
  - Landed: `gg::mx_role` / `mx_gate_up` / `mx_down` (ds4_gguf.cuh) and one dispatch per stage in `moe_gate_up`,
    `moe_down`, `win_experts`, `set_gate_up`, `set_down`; `iq_args` checks only the stages a call uses; the CPU lane's
    two lanes take `gu` / `down` flags and a mixed layer runs one of each (`ent_cap` is the smaller of the two); and
    `expert_layout_mx` now only asks that gate and up match. `ds4_testpack.py` is no longer needed: `pack_gguf.py`
    alone builds the pack the model is.
  - Measured (RX 9070 XT): the real pack loads and runs, layers 26 and 42's i-quant gate/up over the MXFP4 down on
    both the device and the CPU lane, 2200-token prompt at 199 tokens/s. The smoke question still comes out wrong -
    item 4 found the reason elsewhere.

- [x] **The hash layers' routing on the async window path.** Found while comparing with llama.cpp (item 4). The decode
  window publishes its routing through `route_publish` (a mailbox for the CPU lane, and the VRAM experts' plan), and
  that kernel only knew the top-k: on the first `hash_layer_count` layers it took the gate's six best instead of
  `ffn_gate_tid2eid[token]`. The chunk path and the non-async window path always passed the table, which is why every
  earlier run hid it. Now layers 0-2's picks are identical to llama.cpp's (`ffn_moe_topk-0`, tokens 23166 and 3:
  212 61 77 1 133 107 / 102 69 182 162 51 42), and `DS4_TRACE` shows the same six ids on the engine's side.

- [x] **4. Validation against llama.cpp.** Done for the token path: four bugs found, each measured, and the model
  now answers and matches llama.cpp (see "The four bugs" below).  *Left: the same comparison at 8K context, and the
  speed with an expert profile made for this pack.*
  - Then: 8K context, then speed with an expert profile (`tools/dsv41` has the profile tools; the shipped
    `profile-2.00.bin` is for the IST-DAS sets, not this model).

- [x] **5. It is a model Strata installs.** Done in this branch: the family `deepseek4` in `setup.py` (the pin:
  Unsloth's `UD-IQ2_XXS`, three shards, SHA-256, revision `fbbb5b93`), its fit and plan in
  `strata_control/dsv41_fit.py` (`DS4_*`: 43 layers, 256 experts, 11,008 experts, 90.86 GB, 83.87 GB of experts,
  6.99 GB dense, no Engram table; measured on the RX 9070 XT: 701 experts in the card, 11.8 tokens a second), the
  server hand-off (`serve/server.py` and `serve/frontend.py`: `deepseek-v4-flash-0731`, DSML calls, its end token,
  the effort words, the slot environment; the model card's sampling 1.0/1.0), and the launch path (`--family
  deepseek4`, `--model-path` recognising its three pinned shard names, tuning skipped, `--dsv41-package` passed).
  Tests: `tools/test_setup_dsv41.py` (100 pass), `serve/test_dsv41.py`, `tests/terminal/test_launch_dsv41.py`,
  `tests/terminal/test_dsv41_own_files_and_mmproj.py`, `tests/terminal/test_tokenizer_pre.py`.
  - The tokenizer: the GGUF states `joyai-llm`; llama.cpp gives that name DeepSeek's regex sequence
    (`src/llama-vocab.cpp`, the same list as `deepseek3`), and `tools/strata_tokenizer.py` matched only names
    starting with `deepseek`, so it split the text with the Qwen pattern.  Its ids now match `llama-tokenize` on
    the real shard for English, digits, CJK and punctuation.
  - Left: a `strata-dsv41` package in the signed feed with this engine.  The build recipe is
    `packaging/build_dsv41_hip_linux.sh` (below); the release key that signs the manifest is kept offline, and the
    CUDA / Windows twins still need their own package step.

### The engine package (2026-10-25)

`packaging/build_dsv41_hip_linux.sh` builds `strata-dsv41` alone (not beside the Qwen engine), puts it in its own
component folder with the ROCm libraries it loads and the hipBLASLt tables of the archs it was built for, signs the
manifest when `KEY=<feed-key.json>` is given, and writes the archive + `.entry.json` that
`packaging/feed_tool.py build-feed --dsv41 <folder>` takes.  Measured on this box (RX 9070 XT, ROCm SDK in the
Strata 0.1.40.3 venv, hipBLASLt 1.2.0, no table for it in `tools/hip`):

| | |
|---|---|
| package | `strata-dsv41-0.1.50-linux-x86_64-hip-rx9070.tar.gz` (gfx1200 + gfx1201) |
| size | 346 MB (362,820,357 bytes); 1157.9 MiB and 780 files unpacked |
| build | 4.5 minutes on 12 jobs, `STRATA_PORTABLE=ON` and `STRATA_NATIVE_EXPERTS=ON` |
| install | `strata_control.dsv41_engine.install_package` with a throwaway key: signature and every file checked, unpacked in `engine-dsv41/0.1.50-linux-x86_64-hip-rx9070` |
| runs | `env -i .../strata-dsv41 --version` prints `strata-dsv41 0.1.50 build 57d47fb42`; the 16-token smoke question answers "The capital of France is Paris." |

The release key is not on this machine, so the package made here is signed with a test key and cannot be installed by
a PC that trusts the release key.  Signing (`feed_tool.py sign-package`) and publishing the feed remain the owner's
steps.  The Qwen engine's package is built by `packaging/build_hip_linux.sh`, which still carries `strata-dsv41`
beside `strata` for the V4.1 setup; the two are separate artifacts on purpose.

## The four bugs (item 4), and how each was found

The method: run the engine and llama.cpp on the same token (23166, position 0) and compare every tensor either side
writes.  The engine's taps: `DS4_DUMP_FFN=<layer>:<file>` (the FFN's input, the MoE's output, the shared expert's),
`DS4_DUMP_ATTN=<layer>:<file>` (the attention output before `hc_post`), `DS4_DUMP_L=<file>` (every layer's residual
streams, records `[layer, p0, T, T x hc x dim]`), `DS4_TRACE`.  llama.cpp's side: `examples/eval-callback` prints every
graph tensor (`llama-eval-callback -m <shard 00001> -p "Hi!" -n 1 -c 512 -ngl 0 -t 24`; `-ngl 999` tries to put 86 GB on
the card) - its text dump prints 6 values a vector, so for exact numbers a small tool that calls `ggml_backend_tensor_get`
in the same callback writes the tensors as f32 files.  Two traps: the engine's 1-token runs take the *window* path,
prompts longer than a window take the *chunk* path (`DS4_CHUNK_MIN=100000` forces the window one), and each path has its
own dumps.  q and the window's rows needed a tap of their own for bug 1 (a temporary one: `d.q`, `d.kvr`, the row
`ext` holds, the sink).

1. **The per-head q norm was missing.** deepseek4 normalizes q one head at a time after `wq_b`, with no weight
   (llama.cpp's `ggml_rms_norm(q)`).  Engine q vs llama.cpp's: -0.0019 -0.0237 0.0000 against -0.0539 -0.6238 0.0035.
   `Finish::norm` (ds4_kernels.cuh) makes `finish` normalize without a weight; the packer writes `attn_q_head_norm 1`
   for deepseek4 only, so a V4.1 pack keeps the plain `wq_b` output.  After it, q matches to cos 0.99997.
2. **The hyper-connection halves took the wrong pre-mix.** The engine (Maya's `model.py`) lags: the attention reads the
   previous layer's FFN pre-mix and the FFN this layer's attention pre-mix.  deepseek4 does not - each half takes its
   own, as llama.cpp's graph has it.  Layer 0's FFN input was the give-away: cos 0.838 against llama.cpp's `ffn_norm-0`
   (rms 0.228 both), and the router's 256 logits cos 0.842, while the attention output was already 0.9992.  The pack key
   `hc_lag` (0 for deepseek4, absent for V4.1) picks the scheme at four sites (the attention and the FFN, window and
   chunk).  After it, the FFN input is cos 0.9994, its gate/up 0.9998, the shared expert's output 0.9996.
3. **The CPU lane computed nothing for pure i-quant layers.** `cpu_experts_iq`'s guard read
   `if ((gu && !gg::mx_role(l.lay, 0)) || (down && !gg::mx_role(l.lay, 2))) return;` - it returns when the roles are
   *not* MXFP4, i.e. for every i-quant layer, which is every layer of an IQ2_XXS pack.  The experts that were not in
   VRAM contributed nothing (the routed sum equalled the VRAM experts' alone: cos 0.717 against llama.cpp's
   `ffn_moe_out-0`, rms 0.0682 against 0.1023) while the engine counted them as used.  One `!` each side.  After it the
   routed sum is cos 0.9989 and the router's weights match llama.cpp's (0.218 0.139 0.344 0.173 0.337 0.289 against
   0.214 0.140 0.346 0.176 0.335 0.289).
4. **The chunk path never filled the hash layers' token ids.** `d.tids` (which layers 0..`hash_layer_count-1` route by,
   `ffn_gate_tid2eid[token]`) is written by the window path only, and the chunk path's routing passed `d.tids + r0` on a
   `kWin`-entry buffer for slabs of up to `kVlRows` rows.  So a prompt longer than a window routed its first three
   layers by whatever the last window left in that buffer: layer 0's FFN output was cos 0.924 (token 0) and 0.469
   (token 3), and the next token's argmax was wrong (940 against 5211).  `d.tids` now holds a chunk and the chunk path
   uploads each slab's ids.  After it, the chunk path's FFN output is cos 0.9996 / 0.9992 and its logits agree with
   llama.cpp's.

After the four (prompt `23166,3,4,5`, layer 0, tokens 0 and 3): the attention output cos 0.9992-0.9993, the FFN's input
0.9990-0.9995, its output 0.9991-0.9996, the residual streams of layers 0-2 0.9978-0.9998 (both paths), and the next
token's logits cos 0.9971 (window) and 0.9959 (chunk) with llama.cpp's argmax (5211).  The logit differences (rms 0.42
of a spread of about 18) are the engine's FP8 rounding of the window's rows and of the attention output: DeepSeek's
inference rounds them (the reference's `act_quant`), llama.cpp does not.

Speed on this box (UD-IQ2_XXS, 24 CPU threads, 8192 context, no drafter; 701 of 11008 experts in the 16 GiB card's
memory, the rest on the CPU lane).  All after bug 3, so the CPU lane really runs:

| run | tokens | speed | disk read a token | RAM store |
|---|---:|---:|---:|---|
| a build compiling on the same cores | 4096 | 8.17 tokens/s | 125 MB | 48 GB (90.6% hits) |
| the box to itself | 512 | 9.26 tokens/s | 133 MB | 48 GB (88.0% hits) |
| the box to itself, `--ssd-cache 64` | 256 | 9.68 tokens/s | 79 MB | 64 GB (93.2% hits) |

The per-token budget is the CPU lane (76-115 ms, 143 of 258 routed experts a token) and the disk behind it; the layers
themselves are 7.8 ms.  The RAM store is worth sizing: with 64 GB the reads nearly halve.  The store's contents come
from `expert-profile.bin`, which the engine rewrites at the end of every run: a pack whose profile was built from a few
one-token runs holds the wrong experts and streams four times as much, so profile it with a few thousand tokens first.
`--dspark on` is not usable for this model yet: 255 rounds, 0 drafts offered, and it costs 11 ms a round (7.0
tokens/s).  The earlier 33 tokens/s figure was taken while bug 3 was silencing the CPU lane - not a speed this model
can run at.

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
hash table, and the head with its hyper-connection. The compressors are checked against a double reference
(`ds4_compress_test`); everything else is checked against llama.cpp, stage by stage (item 4). The pack is built by
`pack_gguf.py` alone; `ds4_testpack.py` is history. What is left is the 8K-context comparison and an expert profile
made for this pack.

## Cautions

- The `strata-dsv41` package for the signed feed is not published yet; `packaging/build_dsv41_hip_linux.sh` makes
  the Linux AMD one (signed with a test key here), and the release key that makes it installable is offline.
- `strata launch`/`setup.py` know the model (`--family deepseek4`), but a stranger's PC has no engine package to
  download until the feed carries one; there it needs `STRATA_DSV41_EXE` or `--dsv41-package`.
- The test rewrites change values, not shapes: never ship such a pack. `ds4_testpack.py` is kept only in the history.
