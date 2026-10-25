# AGENTS.md

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

## Where this checkout stands (2026-10-25)

This folder is the 0.2.4 tag; the DeepSeek-V4-Flash-0731 port is branch `f-deepseek-v4-flash-0731`, in the
`f-dsv41-ds4` worktree. Read **[AGENTS_run.md](AGENTS_run.md)** (same folder) before anything else: it is the run book -
what is built, how to start DeepSeek, every measurement with the machine it was taken on, and what is open, item by item.

- **DeepSeek-V4-Flash-0731 serves on 8090**, started by `./run-deepseek-v4-flash-0731-ud-iq2_xxs.sh` (or the same
  `serve/server.py` without `--open`). The engine is
  `/mnt/nvme_lexar_0/ollama_user_cache/strata/build-dsv41/strata-dsv41`, 0.1.50 build `4850944e3`, built from the
  `f-dsv41-ds4` worktree on that branch. **This folder cannot run that model** - its packer and
  loader predate the Unsloth `deepseek4` files - so the dsv41 engine code lives and is built in that worktree. Stop the
  server with SIGTERM (a clean stop writes `expert-profile.bin`), and do not relink the engine while it runs.
- **The IQ CPU rows are ported and measured**: `src/dsv41/ds4_iq_cpu.cpp`'s tuned rows (Maya's, behind a shim over this
  build's `gg_cpu_*` lane) compile in, are bit-checked against ggml-cpu's `vec_dot` before any use, and are chosen per
  token count. On this PC they lose, so `own()` declines them and decode is unchanged, bit for bit: 1 token IQ2_XXS 324
  vs 272 µs, IQ3_XXS 465 vs 418; forced on the real model, 7.11 vs 7.95 tok/s with every expert on the CPU lane.
  `DS4_IQ_VERBOSE=1` prints the check, the tier and both measurements; `ds4_iq_own_bench`
  (`tests/dsv41/iq_own_bench.cpp`) is the same comparison without a GPU or the model.
- **What is open**: the AVX-512 IQ3 rows (written, gated, bit-checked, never run on AVX-512 hardware - the comment over
  `iq3_rows_512` says how to settle it in half an hour on such a CPU); the 22 % of a decode the CPU lane spends idle
  (layer overlap, or DSpark's multi-token batches); more experts in VRAM (760 of 11,008 today); and the one-token dot
  itself, which is compute-bound - 53 % of a decode's cycles, and an expert in L3 costs what one streamed from RAM costs
  (4.03 ms).
- **A/B on this box**: `--vram-experts 0 --ssd-cache 96`, 256 tokens, 3 runs each (spread under 1 %). The store's size
  alone is worth about 10 %: 10.80 tok/s at 48 GB against 11.6-12.0 at 96 GB. Another session may be working in the same
  worktree - check `git status` and `ps` there before building, and never kill a run you did not start.
- The commits of this work are listed in AGENTS_run.md's experiment rows; on `f-deepseek-v4-flash-0731` they are the
  whole port, from its first track to the IQ rows, and this page.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For an agent that has a shell, the short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh                  # Linux, Mac
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
