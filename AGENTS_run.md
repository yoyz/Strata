# Running DeepSeek-V4-Flash-0731 on this PC

Date: 2026-10-25. This is the run book for the `deepseek4` port: what is built, how to start it, what was measured,
and what is still open. The engine's own note is `docs/DEEPSEEK4_PORT.md` in the port worktree
`/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4` (branch `f-dsv41-ds4`, commit
`ed032f3d`); this file is about running it from here.

Branch names changed on 2026-10-25: `f-dsv41-iq-own-rows-wt` is now `f-deepseek-v4-flash-0731`, the branch that
worktree has checked out (`f-dsv41-ds4` is the branch it was cut from). The rows below keep the name each was measured
under.

## What is running

| | |
|---|---|
| model | DeepSeek-V4-Flash-0731, Unsloth `UD-IQ2_XXS`, 3 shards, 90.86 GB, revision `fbbb5b93fb787c21338159b0af3318bb3f4d9768` |
| engine | `strata-dsv41` (HIP, gfx1201), local build `/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41/strata-dsv41` (0.1.50 build `4850944e3`, rebuilt 2026-10-25 09:16 from the worktree branch `f-dsv41-iq-own-rows-wt`: `eb6a0d47` the IQ rows ported + `4850944e` their bit check shown; 02:50 was `eb6a0d47` alone, 01:58 the `swaps=` build), `-DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON` |
| pack | `/mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack3` (metadata only: `ds4.txt`, `tensors.txt`, `engram.bin`, `tokenizer/`, `expert-profile.bin`); the 90.86 GB of weights stay in the GGUF shards |
| server | `serve/server.py` from the `f-dsv41-ds4` worktree (the `deepseek4` family: model name, DSML calls, end token, effort words, slots) |
| launcher | `run-deepseek-v4-flash-0731-ud-iq2_xxs.sh` + `strata-deepseek-v4-flash-0731-ud-iq2_xxs.json` in this folder (both gitignored by `.gitignore`: `/run-*.sh`, `/strata-*.json`) |
| PC | Ryzen 9 3900X (12 cores / 24 threads, Zen 2, AVX2), 125 GiB RAM, no swap, RX 9070 XT 16 GiB (gfx1201), one NVMe |

The model's HF folder is
`/mnt/nvme_lexar_0/ollama_user_cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ2_XXS/`.

## Run it

```sh
cd /home/ollama/build/strata-0.2.4-deepseek-v4.1
./run-deepseek-v4-flash-0731-ud-iq2_xxs.sh
```

The script `cd`s to the port worktree (the server code lives there), checks the engine exists, and starts the
server on **8090** with `--open`. 8080 is left alone (Open WebUI uses it). The Qwen 0.2.4.2 launcher in
`~/build/Strata-0.2.4.2` also passes `--port 8090` (its JSON says 8080): run one of the two at a time, or change one
port in its script and JSON.

- Web chat: <http://127.0.0.1:8090/>
- OpenAI: `http://127.0.0.1:8090/v1` (model `deepseek-v4-flash-0731`, any/no API key)
- Anthropic: `http://127.0.0.1:8090/v1/messages`
- Health: `curl -s http://127.0.0.1:8090/health`
- Metrics (one JSON object, the last requests included): `curl -s http://127.0.0.1:8090/metrics | python3 -m json.tool`
  - `requests[].swaps` is the expert cache's moves into VRAM, cumulative since the engine started; the difference
    between two requests is that request's swaps. `null` means an engine older than 2026-10-25 01:58.

Stop it with **SIGTERM** (Ctrl-C in the window, or `kill <server pid>`), not `kill -9`: the clean stop makes the
engine save `expert-profile.bin`. The engine process follows the server.

Over SSH, drop `--open` from the script (there is no browser to open).

## The config, and why

`strata-deepseek-v4-flash-0731-ud-iq2_xxs.json`:

| | | |
|---|---|---|
| `--pack` | `.../ds4-pack3` | the pack's tensor map; the weights are read in place from the shards |
| `--max-context` | `131072` | 128K. The engine's KV is fp4 over compressed rows: 0.02 GB at 8K. **8192 is the only context compared stage-by-stage with llama.cpp**; 32K/128K ran but are not quality-checked |
| `--eos` | `1` | DeepSeek's end token |
| `--dspark` | `none` | the DSpark drafter offers 0 drafts and costs 11 ms a round (`docs/DEEPSEEK4_PORT.md`) |
| `--ssd-cache` | `100` | the RAM store in GB: 10,130 slots of 9.9 MB |
| `host` / `port` | `127.0.0.1` / `8090` | local only; the script passes the same port |
| `backend` | `hip` | the ROCm libraries in `lib_dirs` (the wheel paths) |
| `engine_ready_s` | `3600` | a 91 GB model read from local files; the first start loads for a while |

`--expert-profile` is not in the config: the engine reads `<pack>/expert-profile.bin` by default.

## Measured on this PC

All runs are with the 100 GB store unless said otherwise, one request at a time.

### Start

```
RAM store: 10130 slots of 9.9 MB (100.0 GB), policy 0, 8 reader threads, unbuffered reads
RAM store: 9874 of the profile's hottest experts read in (74.7 GB) in 11.8 s
CUDA0 layers [0, 42] + head: weights 6.77 GB, experts 6.14 GB (760 of 11008 in VRAM), caches 0.02 GB, 588 MiB free
10248 routed experts on the CPU lane (78.3 GB, in place in the model's files), 12 threads
```

- engine RSS: 79 GB; `free`: 84 GB used, 11 GB free, 37 GB cache, **41 GB available**, no swap.
- The 80 GB store started and answered `/health` in about 24 s; the 100 GB start preloaded 74.7 GB in 11.8 s
  (the health time for that start was not recorded).

### A long turn (2026-10-25, `/metrics`)

| | |
|---|---|
| prompt | 25,685 tokens, **6,590 reused** from the previous prompt |
| prefill | 19,095 new tokens at **334 tok/s** = 57.2 s |
| output | 409 tokens at **10.4 tok/s** = 39.2 s |
| total | 96.4 s |
| store | 9,874 of 10,248 CPU-lane experts resident (96 %) |

Earlier 1,049-token turn: 10.9 tok/s, 99 s. A short warm reply (10 tokens) ran at 14.3 tok/s; a very short prompt
reads at only 3.7-5 tok/s because the fixed costs dominate. The port's 256-token `gen.py` check with the 48 GB
store was **11.8 tok/s**.

The 19K prefill was **new content**, not a cache miss: `/metrics` shows `reused: 6590`. A turn that only appends
to a stable prompt reuses its prefix and prefills in seconds.

### What the numbers say

- Decode is CPU-lane bound: 760 of 11,008 experts are in the card (6.14 GB, beside 6.77 GB of dense weights), so
  ~94 % of each token's experts are computed on the 12 CPU cores.
- At 96 % store hits the SSD is nearly out of the decode path (the remaining ~4 % of experts cost ~30-40 MB a token).
- The 100 GB store is the practical maximum here: 41 GB available, no swap. Do not raise it.
- The one-token dot, not RAM, sets the CPU lane: on 12 threads IQ2_XXS rows run at 23 GB/s where a plain read of the
  same bytes reaches 42 GB/s (`ds4_gg_bench --type 16 --mb 2048 --threads 12`), and 24 threads do not help (23.7 vs
  23.0 at 12). A decode's profile agrees: 53 % of its cycles are in the two one-token dots. So a faster one-token dot is
  the only lever left on the CPU lane, and ggml-cpu's is the fastest in the tree here (see the IQ rows experiment).

## Experiments and results

| experiment | result |
|---|---|
| Port correctness vs llama.cpp (`docs/DEEPSEEK4_PORT.md`) | attention cos 0.9992, FFN input 0.9994, FFN output 0.9996, residual streams 0.9978-0.9998, next-token logits cos 0.9959-0.9971 with the same argmax. What is left is the engine's FP8 rounding of window rows (DeepSeek's own inference does it, llama.cpp does not) |
| Item 5: the model Strata installs | family `deepseek4` in `setup.py` (pin, 3 shards, SHA-256, fit), `strata_control/dsv41_fit.py` (`DS4_*`), the server hand-off, the launch path. Commit `ed032f3d` in `f-dsv41-ds4`. `tools/test_setup_dsv41.py`: 100 pass (16 s). Launch + serve + DS4 + tokenizer: 149 pass, 2 skipped (the real-pack tests need `STRATA_DSV41_PACK`/`STRATA_DSV41_MODEL`) |
| Tokenizer `joyai-llm` | llama.cpp gives that name DeepSeek's regex sequence; `tools/strata_tokenizer.py` matched only `deepseek*`, so it split with the Qwen pattern. Fixed; the ids of English, digits, CJK and punctuation match `llama-tokenize` on shard 1 |
| Context 8192 → 32768 → 131072 | ran at all three. 8K is the validated one; 32K/128K have not been compared with llama.cpp. The ratio-128 rows are exact to 65,536 tokens; beyond that is the model's own 1M design, not a checked path |
| Store 48 → 80 → 100 GB | 94.5 % hits at 48 GB (51.6 MB read a token, the 256-token run); an earlier 512-token run before the profile was built read 133 MB a token at 88 %. At 100 GB the store preloads 74.7 GB and holds 9,874 of 10,248 CPU-lane experts (96.4 %). The store's contents come from `expert-profile.bin`; it is rebuilt after a run, so a later start preloads a better set |
| Thinking on/off | with thinking on the model writes a full answer inside ` thinking`, then writes it again after the end of thinking: the 96.4 s turn had 204 thinking and 205 answer tokens. `/thinkingoff` (or `reasoning_effort: "none"`) removes the preamble, the duplicate display and halves context growth |
| CPU lane threads | Linux uses one thread per physical core (`thread_siblings_list`, first entry): 12 threads on the 12 cores; the 12 SMT threads are left to the host/OS. `DS4_CPU_SMT` exists only in the Windows branch. This is the wanted behaviour (physical cores only) |
| IQ2_XXS CPU rows (experiment 1, 2026-10-25) | `DS4_IQ_VERBOSE=1` printed **nothing**: `src/dsv41/ds4_iq_cpu.cpp` starts with `#undef STRATA_NATIVE_EXPERTS`, so Maya's tuned AVX2 rows (IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S, the `g_ct16` one-table form, the gather form, `own()` and its bit check) are **compiled out**. The stub routes to `ds4_gg_cpu.cpp`, which uses `ggml_get_type_traits_cpu(...)->vec_dot` (its own header: "plain and complete, not tuned"). `DS4_IQ_OWN`, `DS4_IQ_GATHER` and `DS4_IQ_VERBOSE` do nothing in this build. The tuned rows are in the tree but need Maya's `strata::dsv41k::cpu` native-expert split-base reference (the `include/strata/kernels/cpu/native_expert.hpp` here is the Qwen one, another API). Porting that, or a small adapter that makes `own()` compare against `gg_cpu_*` / ggml's `vec_dot`, is the one code-side path with a chance of a big win. It was not attempted; no measurement |
| Expert cache, visible | the engine's `DONE` line now ends with `swaps=<n>` (cumulative moves into VRAM); `serve/server.py` reads it and `/metrics` shows it as `requests[].swaps`. The swap rule itself is unchanged. Test: `serve.test_dsv41.DoneLine` |
| The IQ rows ported (experiment 1 continued, 2026-10-25) | Maya's rows **compile in now**: the missing `strata::dsv41k::cpu::native_*` split-base API is a shim over this build's own lane (`gg_cpu_*` = ggml-cpu's `vec_dot`, gate and up as two row arrays, one SwiGLU with the clamp; the trellis types keep their lane through the same shim). So `own()` bit-checks and times the rows against exactly what the engine runs without them, and `DS4_IQ_OWN` / `DS4_IQ_GATHER` / `DS4_IQ_VERBOSE` work. Two changes on top: `own()` now decides **per token count** (1 token is ggml's dot, ≥2 this build's own `iq256x` rows — the lane differs, so the comparison does too), each with the 10 % margin; `DS4_IQ_512=0/1` denies or forces the 512-bit IQ3 rows where the CPU has AVX-512 VBMI, and `DS4_IQ_VERBOSE` names the tier and both measurements. The measuring costs 0.19 s once at load for IQ2_XXS + IQ3_XXS (`ds4_iq_own_bench --experts 1`: 0.21 s against 0.02 s with `DS4_IQ_OWN=0`). Branch `f-dsv41-iq-own-rows-wt` in `f-dsv41-ds4` (commits `eb6a0d47` the port and `4850944e` the bit check's message; engine built 02:50, rebuilt 09:16), the same patches as `f-dsv41-iq-own-rows` in this checkout. The bit-exact test reports itself now: `DS4_IQ_VERBOSE=1` prints, a type at a time, `ds4: CPU lane iq2_xxs rows: bit check against ggml-cpu's dot PASSED (13 rows x 8 blocks, 1/2/3/5/9 tokens, both forms)` (or `FAILED, ggml-cpu's rows stay` - then the rows are not used at all), and `ds4_iq_own_bench` prints the same for the whole pipeline: `bit check: 196608 down-row outputs (48 experts x 1 token(s) x 4096 rows), engine default vs the rows: identical`. New `tests/dsv41/iq_own_bench.cpp` → `ds4_iq_own_bench` (no GPU, no model) runs the engine's decode pipeline on random IQ2_XXS / IQ3_XXS experts. **Result on this PC: the AVX2 rows lose to the engine's lane and `own()` declines them.** 64-row jobs (the engine's `RB`), 12 threads: 1 token IQ2_XXS ours 324 µs vs engine 272 µs, IQ3_XXS 465 vs 418 (the one-table form 365, gather 714); 4 tokens 470 vs 414 and 573 vs 520. Pipeline GB/s 1 token 20.4 vs 20.3 (x1.00), 4 tokens 14.0 vs 14.4 (x0.97), outputs bit-identical. Real model (`--ids 23166 --n 64`, 48 GB store): **10.80 tok/s decode**, unchanged in the A/B (see the next row). The AVX-512 IQ3 rows cannot run here (Zen 2 has no AVX-512) and are not measured — that is where this port can pay |
| The rows forced, the real model (2026-10-25) | `--vram-experts 0` leaves all 10,248 experts on the CPU lane, so the two lanes differ only in the dot; 256 tokens, 96 GB store, 12 threads, 3 runs each, interleaved: **engine lane 7.95 / 7.94 / 7.96 tok/s** (32.03-32.12 s), **Maya's rows forced (`DS4_IQ_OWN=1`) 7.07 / 7.13 / 7.13** (35.74-36.05 s) — **10.6 % slower**, run-to-run spread under 1 %. The default placement is noisier and is not used for this: 760 experts in VRAM gave 11.6-12.0 tok/s on the engine lane and 11.8-12.5 with the rows forced, 10.9-12.5 over all runs, because the store's hits and the VRAM/CPU split move between runs (`CPU lane: 27k-36k of 65,790 experts`). Also on the way: the store's size dominates a decode turn — the 64-token run at 48 GB (89 % hits, 22.5 ms a token waited on disk) made 10.80 tok/s, the 256-token runs at 96 GB (96 % hits) 11.6-12.0. The default stays as it is: `own()` declines the rows here, and the engine computes the same bits at the same speed as before the port |
| Where a decode's cycles go (perf, 2026-10-25) | `perf` on the serving engine (the user's capture): `ggml_vec_dot_iq2_xxs_q8_K` 30.6 % + `ggml_vec_dot_iq3_xxs_q8_K` 22.8 % = **53.4 % in the two one-token dots** - the lane `ds4_iq_own_bench` times, and the one the rows are slower than here; `Pool::loop` 12.8 % + vdso clock 7.7 % + `steady_clock::now` 0.95 + `clock_gettime` 0.57 = **22.0 % in the worker pool's idle spin**; `InterruptSignal::WaitRelaxed` 6.2 % (the GPU); disk and kfd 3.6 %; everything else under 1 %. The 22 % is the CPU lane waiting, on the same 12 cores the workers own, so it is not stealing from the dots - and it is not a fixable overhead: the pool read `steady_clock::now()` on every pause, where the vdso read costs more than the pause it guards, and reading it every 64 pauses instead left decode at **7.85 vs 7.86 tok/s** (256 tokens, `--vram-experts 0`, 96 GB store, 12 threads, 3 runs each: 7.79-7.88 both ways). The spin is time-bounded, so its CPU seconds are the same either way; the patch was reverted, and the tree is as measured. The 53 % is the wall: this CPU's fastest one-token dot is ggml-cpu's own, and the tuned rows cost 10.6 % |
| `/dev/nvidiactl` and `modprobe` | the box had the node but no `nvidia` module; the setup tests reached a real `nvidia-smi`, which made the kernel spawn `modprobe` in a loop. The driver packages were removed, the tests now mock `dsv41_engine.install`, and `Catalog.engine_vendor` honours a known `engine_vendor` instead of probing. No test in that run probed it |

## Not done, and the next experiments

1. **A `strata-dsv41` package in the signed feed with this engine.** `packaging/package_engine.py --require-dsv41`
   already takes it; until a package exists, setup finds no engine for `deepseek4` unless one is named
   (`STRATA_DSV41_EXE` or `--dsv41-package`). This is the one thing missing before a stranger can install it.
2. **The 8K-context comparison against llama.cpp** (the same method as the token path), then speed with a shipped
   expert profile for this pack.
3. **DSpark**: 0 drafts offered; the drafter blocks need the same kind of port.
4. **A/Bs worth measuring**, in order:
   - `--mmap-experts off` (all 78 GB of CPU experts in RAM, no store lookups or misses) against the 100 GB store.
   - The IQ2_XXS own rows **are compiled in now** (see the experiment row) and this PC's own measurement declines them
     at 1 and 4 tokens. The form A/Bs that are left need a Zen 4 / Zen 5 (AVX-512 VBMI) box, or an Intel one:
     `DS4_IQ_VERBOSE=1` prints both token counts and the tier, `DS4_IQ_512=0/1` picks the 512-bit IQ3 rows,
     `DS4_IQ_GATHER=1` the `vpgatherdd` grid form, `DS4_IQ_AVX2=1` the AVX2 rows on an AVX-512 CPU.
   - `--ssd-policy lfu` and `--prefill-chunk 4096` (cheap).
5. **Quality at 32K/128K** (repetition/drift late in a long prompt), and a `/compact` habit before the top.
6. **An AMD Instinct MI100 (`gfx908`) build is not supported and was not attempted.** Recorded here so it is not lost:
   - The owner reports that the Qwen3.8-Flash-Next **gfx906** build (`-DSTRATA_HIP_GFX906=ON`) succeeded in October
     2026. That build compiles the main engine's CUDA sources through `include/strata/platform/hip_compat/`, which
     maps a CUDA warp to a logical half of the 64-lane wavefront (32-wide shuffles, a ballot of its own half).
   - `strata-dsv41` is explicitly left out of it: `CMakeLists.txt:801` is
     `if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)`. The dsv41 HIP path is wave32 WMMA (gfx11/gfx12; `tc_ok()`
     accepts only those, otherwise the F32 kernels), and its kernels assume 32 lanes.
   - A gfx908 port would mean: adding `src/dsv41` to the wave64 build, auditing its 32-lane math, a ROCm that still
     ships gfx908 code objects (the ROCm 6.x era; hipBLASLt has no gfx908 kernels), a `strata-dsv41` package, then
     `ds4_compress_test`, the 16-token smoke and the logits comparison against llama.cpp.
   - MI100 has 32 GB HBM2. That would hold about 3,000 of the 11,008 experts beside the 6.77 GB of dense weights
     (against 760 today), but the CPU lane still sets the speed. The Radeon AI PRO R9700 (gfx1201) remains the
     supported 32 GB card.

7. **The CPU lane is idle 22 % of a decode** (the profile row): its work waits on the GPU side between layers, on the
   same cores, so no pool or kernel tweak reaches it. Using that time needs layer N's expert rows overlapped with layer
   N+1's attention, or multi-token batches (DSpark, item 3) so the four-token rows (x2.9 against ggml's dot) apply to
   decode. Both are engine work, not a row.

## Engine commands (from the port worktree)

```sh
W=/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4
B=/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41
V=/mnt/nvme_lexar_0/ollama/build/Strata-0.1.40.3/.venv/lib/python3.14/site-packages
M=/mnt/nvme_lexar_0/ollama_user_cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ2_XXS
P=/mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack3
export LD_LIBRARY_PATH=$V/_rocm_sdk_devel/lib:$V/_rocm_sdk_devel/lib/rocm_sysdeps/lib:$V/_rocm_sdk_libraries_gfx120X_all/lib
cd $W

# build (the build dir is configured; see docs/DEEPSEEK4_PORT.md for the cmake line)
HIP_PLATFORM=amd ninja -j 12 -C $B strata-dsv41 ds4_compress_test
$B/ds4_compress_test            # "all compressors match"

# a pack is metadata only; rebuild after a packer change (layer 26/42 need the MXFP4-beside-i-quant layout)
python3 tools/dsv41/pack_gguf.py --model $M --out $P

# the 256-token check: the answer, the engine's speed lines, "the answer names Paris"
python3 tools/dsv41/gen.py --exe $B/strata-dsv41 --pack $P --tokens 256

# the IQ2_XXS/IQ3_XXS CPU-row choice on this CPU
DS4_IQ_VERBOSE=1 $B/strata-dsv41 --pack $P --ids 23166 --n 1 --max-context 8192 --dspark none   # loads the model once

# the same rows without the model: the engine's decode pipeline on random experts, the two lanes and the bits
$B/ds4_iq_own_bench --nt 1 --experts 48 --threads 12
DS4_IQ_VERBOSE=1 $B/ds4_iq_own_bench --nt 1 --experts 48 --threads 12   # the timings own() decides from (1 and 4 tokens)
$B/ds4_iq_own_bench --mode own --nt 1       # ours forced (DS4_IQ_OWN=1), against the same lane; DIFFERENT BITS = a bug

# the real model, the two lanes: 64 tokens, twice each (the store's hits move between runs)
for m in 0 1; do DS4_IQ_OWN=$m $B/strata-dsv41 --pack $P --ids 23166 --n 64 --max-context 8192 --dspark none \
  --ssd-cache 48 --cpu-threads 12; done
```

A real pack never needs `tools/dsv41/ds4_testpack.py` (that was for the pre-`332c09c2` test pack). `--dspark off`
loads the drafter and takes VRAM; `none` neither loads it nor runs speculative rounds.

## Cautions

- The launcher and its JSON are gitignored; the code is committed in `f-dsv41-ds4` (`ed032f3d`). This checkout
  (`strata-0.2.4-deepseek-v4.1`) has the launchers but **not** the `deepseek4` server family: run the script,
  which `cd`s to the port worktree.
- 32K/128K are not the validated context; a long prompt can drift. 8K was the compared one.
- The model is 2-bit: it can repeat itself and it thinks by default. `/thinkingoff` is the quick-chat setting.
- No swap on this PC. 100 GB store + 79 GB RSS is the ceiling; close other large programs before raising it.
- Clean stop (SIGTERM) writes `expert-profile.bin`; `kill -9` does not, and the old profile stays (it is only a
  hot-expert hint, never model data).
