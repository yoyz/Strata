# AGENTS.md

> **Work in progress on this branch: the DeepSeek-V4-Flash-0731 engine (`deepseek4`).**
> This section describes the branch `f-dsv41-ds4` only; the project's own instructions follow it, unchanged.
> Delete this section when the port lands on `main`.
>
> Branch `f-dsv41-ds4`, worktree `/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4`.
> The port's own note - every measurement, how each was taken, and the details behind this summary:
> [docs/DEEPSEEK4_PORT.md](docs/DEEPSEEK4_PORT.md).

## Where this stands (2026-10-25)

`strata-dsv41` runs Unsloth's `UD-IQ2_XXS` of DeepSeek-V4-Flash-0731 (`deepseek4`: 43 layers, 256 routed experts of
which 6 are used for every token, 3 hash-routed layers, compress ratios 0/4/128, an indexer, 4 hyper-connections,
no Engram table) and **answers correctly**: "The capital of France is Paris."  Its pack is built by
`tools/dsv41/pack_gguf.py` alone - no test rewrites.

Against llama.cpp's own graph on the same pack (`examples/eval-callback`, a 4-token prompt, every stage compared):
the attention output cos 0.9992, the FFN's input 0.9994, its output 0.9996, the residual streams 0.9978-0.9998, and
the next token's logits cos 0.9959-0.9971 with **the same argmax**.  What is left over is the engine's FP8 rounding of
the window rows and of the attention output: DeepSeek's own inference rounds them, llama.cpp does not, so it is a
reference difference of 2-5 % of a vector's L2, not a bug.

Speed on this box (RX 9070 XT 16 GiB, 128 GiB RAM, one NVMe, 24 CPU threads, `--dspark none`, 8192 context; 701 of
the 11008 experts in the card's memory and the rest on the CPU lane; the RAM store at its default 48 GB, fed by a
profile built from a few thousand tokens):

| run | tokens | speed | read from the disk a token | RAM store hits |
|---|---:|---:|---:|---:|
| `tools/dsv41/gen.py`, the box to itself | 256 | **12.7 tokens/s** | 40 MB | 95.5 % |
| the same, before the profile was built | 512 | 9.3 tokens/s | 133 MB | 88.0 % |
| a C++ build on the same 24 cores | 4096 | 8.2 tokens/s | 125 MB | 90.6 % |

The per-token budget is the CPU lane (about 111 of the 258 routed experts a token) and the disk behind it; the 43
layers' kernels are 7.8 ms of it.  `--dspark on` is not usable for this model yet: 255 rounds, **0 drafts offered**,
11 ms a round.

## Start here: build, pack, test, measure

```sh
W=/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4
B=/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41
V=/mnt/nvme_lexar_0/ollama/build/Strata-0.1.40.3/.venv/lib/python3.14/site-packages
M=/mnt/nvme_lexar_0/ollama_user_cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ2_XXS
P=/mnt/nvme_lexar_0/ollama_user_cache/strata/ds4-pack3
export LD_LIBRARY_PATH=$V/_rocm_sdk_devel/lib:$V/_rocm_sdk_devel/lib/rocm_sysdeps/lib:$V/_rocm_sdk_libraries_gfx120X_all/lib
cd $W

# 1. build (the build dir is configured already; the cmake line is in docs/DEEPSEEK4_PORT.md)
HIP_PLATFORM=amd ninja -j 12 -C $B strata-dsv41 ds4_compress_test
$B/ds4_compress_test          # the compressors against their double reference: "all compressors match"

# 2. the pack: 3 MB of metadata, the 91 GB of weights stay where they are.  A pack built before 332c09c2 lacks
#    hc_lag 0 and gives wrong output - re-pack.  The expert profile is not the packer's: the engine writes
#    <pack>/expert-profile.bin itself at the end of every run (keep it; re-packing into the same folder leaves it).
python3 tools/dsv41/pack_gguf.py --model $M --out $P

# 3. does it answer?  the smoke question, 16 tokens, the text is read
python3 tools/dsv41/smoke.py --exe $B/strata-dsv41 --pack $P --tokens 16

# 4. the speed and the answer: 256 tokens of the same question
python3 tools/dsv41/gen.py --exe $B/strata-dsv41 --pack $P --tokens 256
#   -> The capital of France is Paris.
#      prompt: 17 tokens in 3132 ms (5.43 tok/s, windows of 16)
#      decode: 256 tokens in 19812 ms (12.87 tok/s); CPU lane: 28343 of 65790 experts; ...
#      store: 95.5% of the CPU lane's experts were in RAM ...; read 38.3 MB a token
#      the answer names Paris
```

`gen.py` sets the engine's `DS4_NO_EOS=1` (so the 256 tokens are really generated; `--stop-at-eos` stops at the
model's end), shows the answer up to the first EOS, prints the engine's own `prompt:` / `decode:` / `store:` lines,
and exits 1 when the reply does not name Paris.  Also: `--ssd-cache N` (the RAM store in GB), `--dspark none|auto`,
`--text "..."` (continue plain text, no question, no check), `--question "..."`.

The GPU must be free (stop the Qwen server first).  Measure on an idle box.  The first run of a fresh pack is slow
and streams about four times as much: the store is filled from `expert-profile.bin`, which the engine rewrites at the
end of every run - profile it once with a few thousand tokens, then the table above.

## The next step

**Item 5: make it a model Strata installs.**

1. `setup.py` / `strata_control`: the family (`deepseek4`), the download pins (unsloth's `UD-IQ2_XXS`, its 3 shards,
   SHA-256), the sizes for `--plan` and `--check` (`strata_control/dsv41_fit.py` knows the V4.1 layout, not this
   one: 11008 experts, 6 a token, no Engram table), and the run's settings (8192 context, `--dspark none` until the
   drafter works).
2. The server hand-off: `serve/server.py` runs `strata-dsv41` for V4.1 by engine name; this model needs its family
   entry (name, template, thinking on by default) and a run of `serve/test_dsv41.py` and `serve/test_family_name.py`.
3. A `strata-dsv41` package for the signed feed (`packaging/package_engine.py`, `packaging/feed_tool.py`, the release
   notes).  None exists for this model: today it is a source build.
4. Then, in any order: the DSpark drafter (it offers 0 drafts - the drafter blocks need the same kind of port), the
   8K-context comparison against llama.cpp, and a shipped expert profile for this pack (the shipped
   `data/dsv41/profile-2.00.bin` is for the IST-DAS V4.1 sets).

## How to continue (the method that found the four bugs)

Compare every stage against llama.cpp, same pack, same tokens:

- llama.cpp's side: `examples/eval-callback` from `third_party/llama.cpp` (`-ngl 0` - `-ngl 999` tries to put 86 GB on
  the card; it needs the ROCm SDK's `LD_LIBRARY_PATH`).  It prints every graph tensor, 6 values a vector; for exact
  numbers, a small program that calls `ggml_backend_tensor_get` in the same callback writes them as f32 files.
- The engine's side: `DS4_DUMP_FFN=<layer>:<file>` (the FFN's input, the MoE's output, the shared expert's),
  `DS4_DUMP_ATTN=<layer>:<file>` (the attention output before `hc_post`; both paths), `DS4_DUMP_L=<file>` (every
  layer's residual streams, records `[layer, p0, T, T x hc x dim]`), `DS4_TRACE=<file>` (the router's picks),
  `--dump-logits FILE` (the next token's logits - the end-to-end check).
- Traps: a 1-token run takes the *window* path, a prompt longer than a window takes the *chunk* path
  (`DS4_CHUNK_MIN=100000` forces the window one), and the two are separate code - a fix in one is not a fix in the
  other (bug 4 below was chunk-only; bugs 2 and 3 were in both).  `--window 16` sizes the prompt windows.

## The four bugs, one line each (numbers in the port note)

1. The per-head RMS norm on `q` after `wq_b` (llama.cpp's weightless `ggml_rms_norm(q)`) was missing - all of q was
   off by a per-head scale.  Pack key `attn_q_head_norm`, `Finish::norm`.
2. The hyper-connection pre-mix was V4.1's *lagged* one (the attention reads the previous layer's FFN pre-mix);
   deepseek4 takes each half's own, as llama.cpp does.  Pack key `hc_lag` (0 for deepseek4) - without it the FFN's
   input was cos 0.838 against llama.cpp's.
3. `cpu_experts_iq`'s guard was inverted (`if ((gu && !mx_role) ...) return`), so every pure i-quant layer's CPU lane
   returned before computing anything: the experts not in VRAM contributed nothing while the engine counted them as
   used (the routed sum was the VRAM experts' alone).
4. The chunk path never filled `d.tids` - the ids the three hash-routed layers route by - and passed `d.tids + r0`
   into a 16-entry buffer, so a prompt longer than a window routed layers 0-2 by a stale window.

## Files this branch creates or changes

Created:
- `docs/DEEPSEEK4_PORT.md` - the port's note: status, the method, the four bugs with their numbers, the speed table,
  the model's geometry, the packer's keys, the restart commands.
- `tests/dsv41/compress_test.cpp` - the compressors (CSA and HCA, the position table, a group across chunk
  boundaries) against a double written from llama.cpp's graphs; the `ds4_compress_test` target.
- `tools/dsv41/gen.py` - N tokens of the smoke question: the answer, the engine's speed lines, a check that it names
  Paris.  `--tokens 256` is the check this file asks for.
- `tools/dsv41/ds4_testpack.py` - obsolete, kept in the history: the scripted test pack from before a layer could
  carry MXFP4 beside i-quant gate/up.  A real pack never needs it.

Changed (the `deepseek4` support from the first commit `335fcfa1` to the four fixes in `332c09c2`):
- `src/dsv41/ds4_model.cu` - the port itself: the head's hyper-connection, the hash layers' routing, the compressor
  (two halves a row, the position table, the ring that keeps a group across a window), the indexer's own compressor
  (the LID keys) and its Hadamard rotation, every compressed row on the ratio-128 layers, the per-head q norm,
  `hc_lag`, the CPU lane's i-quant guard, the chunk path's token ids, the MXFP4-beside-i-quant dispatch, and the
  debug taps above.
- `src/dsv41/ds4_kernels.cu` and `src/dsv41/ds4_kernels.cuh` - `compress_kernel` / `compress_win` (the ratio-4 and
  ratio-128 compressors), `hadamard_rows`, `index_score` over 64 heads, `Finish::norm`, and `finish` normalizing
  without a weight.
- `src/dsv41/ds4_gguf.cu` and `src/dsv41/ds4_gguf.cuh` - one dispatch a stage for the routed experts
  (`mx_role` / `mx_gate_up` / `mx_down` / `expert_type`), which is what lets a layer carry IQ2_XXS gate/up over an
  MXFP4 down (layers 26 and 42 of this pack).
- `tools/dsv41/pack_gguf.py` - accepts `deepseek4`; maps `attn.compressor.ape`, `attn.indexer.compressor.*` and the
  head's `hc_*`; carries `ffn_gate_tid2eid` and `hash_layer_count`; writes `kv_source_layer_ids`,
  `index_source_layer_ids`, `attn_q_head_norm` and `hc_lag`; a header-only `engram.bin` when the model has none; the
  tokenizer's `pre` from the GGUF.
- `CMakeLists.txt` - the `strata-dsv41` and `ds4_compress_test` targets under `STRATA_BUILD_DSV41`.

Untouched on this branch, and needed by item 5: `setup.py`, `strata_control/`, `serve/`, `packaging/`.

## Gotchas

- A pack is metadata only; the 91 GB of weights are read in place from the GGUF.  Re-pack after any packer change.
- `--dspark none` does not load the drafter but leaves speculative rounds on (one token a round, ~0 cost); `off`
  loads the drafter and its bytes take video memory; `auto` decides by confidence (this model gets 0 drafts).
- The engine stops at EOS (token 1) right after the answer - that is why a 256-token `--tokens` run needs
  `DS4_NO_EOS=1` (gen.py sets it).
- `strata-dsv41 --pack-info` prints a pack without touching a card.
- A build on the same 24 cores takes about a third off the token rate (8.2 against 12.7 tokens/s): measure idle.

## First message for the next session

> Continue the DeepSeek-V4-Flash-0731 port (`deepseek4`) on branch `f-dsv41-ds4`, worktree
> `/mnt/nvme_lexar_0/ollama/.local/share/opencode/worktree/cfd61a/f-dsv41-ds4`.  Read the section at the top of
> `AGENTS.md` and `docs/DEEPSEEK4_PORT.md`, then run the 256-token check (`tools/dsv41/gen.py`, the "Start here"
> block) to confirm the state.  The next step is item 5 there: the family in `setup.py` and `strata_control` (pin,
> sizes, fit), the server hand-off, and a `strata-dsv41` package for the feed.


Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5, Unsloth, Gyro and NVFP4
variants, and Qwen3.6-35B-A3B) on a normal PC: one NVIDIA or AMD graphics card plus system RAM on Windows or Linux,
or an Apple Silicon Mac. Version 0.2.3, engine 0.1.48. It has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. The installer has the engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is the older one-click installer from source.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For an agent that has a shell, the short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). The old route (clone
the repository and run `START-HERE.bat --yes` or `./setup.sh --yes`) still works and is the fallback when the
Terminal cannot run. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`). The app's UI
  tests run with `npm test` in `apps/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
