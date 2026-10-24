# Porting the DeepSeek engine to DeepSeek-V4-Flash-0731 (`deepseek4`)

**Status: not usable yet — it loads and decodes, with the wrong math.** Measured on 2026-10-24, RX 9070 XT (16 GiB,
gfx1201), 125 GiB RAM, one NVMe, the Qwen server stopped: a pack of the IQ2_XXS set (91 GB of weights) loads in
12.6 s, 765 of 11,008 experts go to VRAM and the rest read in place from the files, and a 3-token prompt decodes at
2.2 tokens/s (1.4 GB read from disk a token; no expert profile, cold store). The tokens are wrong, as expected: two
test-only substitutions were needed to get this far (below), and the four features in this note are still missing.

The two substitutions in the test pack (never in a real one): layer 26's and layer 42's down experts are MXFP4 in
this quant and the engine cannot mix MXFP4 with the I-quants, so `tensors.txt` points them at layer 25's and layer
41's down (same shape, wrong values); and the ratio-4 compressor matrices were cut to their first 512 rows (the engine
expects `head` rows, the model has `2·head`). Both are undone by the real port.

DeepSeek-V4-Flash-0731 (`deepseek4`, 43 layers, 256 experts, no Engram) carries four pieces of the model the engine
(`deepseek41`, 40 layers, 384 experts, an Engram table) has no code for. They are listed below with the reference
needed to write them.

The reference implementation is llama.cpp's `src/models/deepseek4.cpp` in the pinned checkout
(`third_party/llama.cpp`, the one the CUDA/HIP builds use). It reads the same GGUFs and was checked against the same
model, so its math is the target. Nothing in this note is measured on a card yet.

## What the model is

From the GGUF's own metadata (`UD-IQ2_XXS`, 3 shards, 90.9 GB of weights + a 10.9 GB DSpark drafter):

| | |
|---|---|
| architecture | `deepseek4` (llama.cpp's id) |
| layers | 43 |
| experts | 256 routed, 6 a token, 1 shared; `hash_layer_count` 3 |
| hidden / heads | 4096 / 64, head 512, rope 64, q_lora 1024, o_lora 1024 × 8 groups |
| hyper-connections | 4 (the layer `hc_*` tensors the engine already reads) |
| compress ratios | `0,0,4,128,4,128,…,4` (43 entries), then 3 more for the drafter |
| indexer | 64 heads × 128, top-k 512, on the 21 ratio-4 layers |
| context | 1,048,576 (the engine would be checked at 8192, as for V4.1) |
| quant | UD-IQ2_XXS: IQ2_XXS / IQ3_XXS experts, one layer IQ2_S + MXFP4, Q4–Q6_K dense |

The engine's `ds4.txt` already carries all of that after `tools/dsv41/pack_gguf.py` translates the `deepseek4.*` keys
(the packer now does; see “packer changes”).

## The four missing pieces

### 1. The compressor is per layer, with a learned position table

V4.1: a few “source” layers own the compressed KV of a ratio, and the layers after them read it. The engine's
`kv_source_layer_ids` model.

V4.1-Flash-0731: **every** layer with `ratio > 0` carries `attn_compressor_kv/gate/norm/ape` and owns its own
compressed rows. The packer therefore writes `kv_source_layer_ids = [2..42]`, one source a layer.

Two shapes, taken from llama.cpp (`coff = ratio == 4 ? 2 : 1`):

- **ratio 4 (CSA)**: `wkv`/`wgate` are `[dim, 2·head]` = `[4096, 1024]`; `ape` is `[2·head, 4]` = `[1024, 4]`.
- **ratio 128 (HCA)**: `wkv`/`wgate` are `[dim, head]` = `[4096, 512]`; `ape` is `[head, 128]` = `[512, 128]`.

The compressor's score is `wgate(x) + ape[slot]`, where `slot` is the position inside the group (0 … ratio-1). Then,
for each group, softmax over the `coff · ratio` scores and a weighted sum of the `coff · ratio` values, each `head`
wide; RMS norm with `attn_compressor_norm`; RoPE at the compressed positions with `compress_rope_freq_base`
(160000). See `build_overlap_compressed_kv_from_state` (ratio 4) and `build_hca_compressed_kv_from_state`
(ratio 128).

The engine's `compress_win` (`src/dsv41/ds4_kernels.cu:1946`) does the softmax and the weighted sum over `ratio`
rows of `head` values and has no position table, and its per-layer state (`Layer::c_kv`, `c_score`) is `ratio · head`
floats. The ratio-4 layers need `2 · ratio · head`, and the kernel needs the APE add.

### 2. The indexer compresses its own keys

V4.1: the indexer projects the main compressed rows with `indexer.wk` into index keys (`i_wk` on the latent), and an
optional candidate-block step picks blocks.

V4-Flash: the 21 ratio-4 layers carry `indexer_compressor_ape/kv/gate/norm` — a second compressor at 128 wide
(`wkv`/`wgate` `[4096, 256]`, `ape` `[256, 4]`, `norm` `[128]`) — and the indexer's keys are those compressed rows.
`indexer_compressor_norm` is 128 wide (the engine's index head dim is also 128). The query path is as now:
`indexer.attn_q_b` on the q-lora vector, RoPE, `indexer.proj` weights, top-k 512.

So `attn.indexer.wk` and `attn.indexer.k_norm` do not exist in this model; the keys come from a compressor of their
own. See `build_lid_top_k` and the `lid_state_*` path in `build_attention`.

### 3. The first three layers route by token id

`hash_layer_count = 3`, and layers 0–2 carry `ffn_gate_tid2eid.weight`, an **I32** table `[6, 129280]`: for a token
id, the six experts it uses. llama.cpp uses it instead of the top-k of the gate:

```cpp
selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
exp_probs_b = nullptr;                 // no learned router bias on these layers
```

The weights are still the gate's scores at those six experts. The engine's `route()` always takes the top-k of
`gate + bias` (`src/dsv41/ds4_kernels.cu:458`); the table is skipped by the packer today (ggml type 26 is I32, and
`pack_gguf.py` leaves non-weight types out).

The engine also requires `ffn.gate.bias` on every layer; the three hash layers have none. `load_dense` now makes a
zero bias instead (zeros route exactly like “no bias”).

### 4. The head mixes the hyper-connections once more

`output_hc_fn` `[4·dim, 4]`, `output_hc_base` `[4]`, `output_hc_scale` `[1]`: flatten the 4 states to `4·dim`, RMS
norm, one linear to 4 values, `sigmoid(scale·mix + base) + eps`, weighted sum over the 4 states, then `output_norm`
and the head. llama.cpp: `build_hc_head`. The engine applies hyper-connections only inside layers
(`hc_mix_chunk` / `hc_split` / `hc_post`); at the head it goes straight to `norm.weight` and `head`.

## Packer changes (done)

`tools/dsv41/pack_gguf.py`:

- accepts `deepseek4` as a model architecture (`MODEL_ARCHS`);
- maps the new tensors: `attn.compressor.ape`, `attn.indexer.compressor.{ape,wkv,wgate,norm}`, `head.hc_*`,
  `ffn.gate.tid2eid` (the last still skipped as metadata);
- writes a header-only `engram.bin` when the model has no Engram (the engine opens it unconditionally);
- takes the tokenizer's `pre` from the GGUF (`joyai-llm`) instead of assuming `deepseek3`;
- synthesises `kv_source_layer_ids` (every ratio>0 layer) and leaves `index_source_layer_ids` empty (the old
  `indexer.attn_k` is absent; see piece 2).

A pack of the 0731 IQ2_XXS set: `strata-dsv41 --pack-info` reports 43 layers × 256 experts, 6 used; experts
IQ2_XXS/IQ2_S/MXFP4 per layer.

## What the engine still needs

| Where | What |
|---|---|
| `ds4_kernels.cu` `compress_win` / `compress_kernel` | `coff` (2 for ratio 4), the `ape` add by slot, and the 2-row overlap the ratio-4 groups use |
| `ds4_model.cu` `Src` / `Layer::c_kv`, `c_score`, `latent` | ratio-4 rows at `2·head`; the compressed state stays per layer (already the case when each ratio>0 layer is a source) |
| `ds4_model.cu` indexer | a second compressor (128 wide) and its keys, replacing `indexer.wk` on these layers |
| `route()` / `moe_route` | expert selection from `tid2eid[token]` on layers 0–2, gate weights at those ids |
| head | the `output_hc_*` mix before `norm.weight` |
| `pack_gguf.py` + engine | carry the I32 `ffn_gate_tid2eid` table into the pack (today it is skipped) |

Until all four are in, a pack of this model loads but cannot answer correctly: the compressed rows, the attention
selection and three layers' experts would be wrong.
