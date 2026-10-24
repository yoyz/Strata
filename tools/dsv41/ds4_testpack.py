#!/usr/bin/env python3
# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT); part of the deepseek4 port track
# (docs/DEEPSEEK4_PORT.md).  TEST ONLY - not a product tool and never used by setup or the app.
"""Make a test pack of a deepseek4 model for the engine as it stands today.

    python tools/dsv41/ds4_testpack.py PACK OUT

One rewrite, undone by the port and making the pack numerically wrong:

1. layers 26 and 42 of unsloth's UD-IQ2_XXS set have MXFP4 down experts and the engine cannot mix MXFP4 with the
   I-quants; their down matrices are pointed at layer 25's / 41's (same shape, wrong values).

The compressor's shapes are not touched any more: the engine reads deepseek4's two halves a row and the position
table (docs/DEEPSEEK4_PORT.md, item 1).  The output is a copy: the source pack is not changed.  A real pack must
never be built this way.
"""
import pathlib
import re
import shutil
import sys


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    if dst.exists():
        sys.exit("%s: already there" % dst)
    shutil.copytree(src, dst)
    p = dst / "tensors.txt"
    lines = p.read_text().splitlines()

    def down_of(il):
        out = {}
        for ln in lines:
            m = re.match(r"t layers\.%d\.ffn\.experts\.(\d+)\.w2\.weight (.*)" % il, ln)
            if m:
                out[int(m.group(1))] = m.group(2)
        return out

    down_25, down_41 = down_of(25), down_of(41)
    new, n_down = [], 0
    for ln in lines:
        m = re.match(r"(t layers\.(26|42)\.ffn\.experts\.)(\d+)(\.w2\.weight )(.*)", ln)
        if m:
            new.append(m.group(1) + m.group(3) + m.group(4) +
                       (down_25 if m.group(2) == "26" else down_41)[int(m.group(3))])
            n_down += 1
            continue
        new.append(ln)
    p.write_text("\n".join(new) + "\n")
    print("%s: %d down matrices replaced (test only)" % (dst, n_down))
    return 0


if __name__ == "__main__":
    sys.exit(main())
