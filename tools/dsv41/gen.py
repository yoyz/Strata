"""256 tokens (or --tokens) of a question through strata-dsv41: the text, the engine's speed, and whether it answered.

    python tools/dsv41/gen.py --exe strata-dsv41 --pack <pack> [--tokens 256] [--gpus 0] [--question "..."]
    python tools/dsv41/gen.py --exe strata-dsv41 --pack <pack> --text "Once upon a time"    (no question, no check)

The question is the smoke one (the capital of France): the answer is read, not just the exit code, and the engine's own
`prompt:` / `decode:` lines are printed so a run's speed (tokens/s) is there to compare.  Exit 0 when the reply is
sensible, 1 when it is not, 2 when the engine did not run at all.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from dsv41.smoke import PROMPT, sensible, tokenizer  # noqa: E402


def main() -> int:
    for st in (sys.stdout, sys.stderr):       # DeepSeek's special tokens are fullwidth: a Windows console code page cannot print them
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--gpus")
    ap.add_argument("--question", default=PROMPT, help="the question (default: the smoke one, the capital of France)")
    ap.add_argument("--text", help="plain text to continue instead of a question (no answer check, --tokens applies)")
    ap.add_argument("--dspark", default="none", help="none (the drafter offers no drafts for this model yet) or auto")
    ap.add_argument("--ssd-cache", type=int, default=0, help="the RAM store in GB (0: the engine's own size)")
    ap.add_argument("--stop-at-eos", action="store_true", help="stop when the model ends its answer (default: run the "
                                                                  "--tokens asked for, DS4_NO_EOS=1)")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    pack = Path(a.pack)
    tok = tokenizer(pack)
    prompt = a.text if a.text is not None else a.question
    ids = tok.encode(prompt, parse_special=True)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ids.txt"
        f.write_text(",".join(str(i) for i in ids), encoding="ascii")
        cmd = [a.exe, "--pack", str(pack), "--ids", "@" + str(f), "--n", str(a.tokens),
               "--max-context", str(a.context), "--dspark", a.dspark, "--window", "16"]
        if a.ssd_cache > 0:
            cmd += ["--ssd-cache", str(a.ssd_cache)]
        if a.gpus:
            cmd += ["--gpus", a.gpus]
        env = dict(os.environ)
        if not a.stop_at_eos:
            env["DS4_NO_EOS"] = "1"   # the engine's switch: decode --tokens tokens whatever (the answer ends early)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=a.timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"the engine did not finish the run: {e}")
            return 2
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        print(f"the engine stopped with an error (exit {r.returncode}): " + " | ".join(err[-3:]))
        return 2
    last = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
    out = [int(x) for x in re.findall(r"-?\d+", last)]
    eos = 1
    for line in (pack / "ds4.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("eos_token_id "):
            eos = int(line.split()[1])
    answer = out[:out.index(eos)] if eos in out else out   # (a --tokens run goes on past the model's end: show the answer)
    print(tok.decode(answer).strip())
    print()
    for line in (r.stderr or "").splitlines():   # the engine's own speed lines (and the store's hits)
        if line.startswith(("prompt:", "decode:", "decode reads", "store:", "dspark")):
            print(line)
    if a.text is not None:
        return 0
    good, why = sensible(tok.decode(answer))
    print("the answer " + ("names Paris" if good else f"is not sensible: {why}"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
