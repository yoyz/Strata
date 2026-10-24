#!/usr/bin/env python3
# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT); part of the deepseek4 port track
# (docs/DEEPSEEK4_PORT.md).  TEST ONLY - not a product tool and never used by setup or the app.
"""Make a test pack of a deepseek4 model for the engine as it stands today.

    python tools/dsv41/ds4_testpack.py PACK OUT

Two rewrites, both undone by the port and both making the pack numerically wrong:

1. layers 26 and 42 of unsloth's UD-IQ2_XXS set have MXFP4 down experts and the engine cannot mix MXFP4 with the
   I-quants; their down matrices are pointed at layer 25's / 41's (same shape, wrong values).
2. a ratio-4 layer's compressor matrices are [2*head, dim] (llama.cpp's coff = 2), the engine's compressor still
   reads [head, dim]; only the first half is kept.

The output is a copy: the source pack is not changed.  Remove both rewrites when the port in docs/DEEPSEEK4_PORT.md
is done; a real pack must never be built this way.
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
    new, n_down, n_cut = [], 0, 0
    for ln in lines:
        m = re.match(r"(t layers\.(26|42)\.ffn\.experts\.)(\d+)(\.w2\.weight )(.*)", ln)
        if m:
            new.append(m.group(1) + m.group(3) + m.group(4) +
                       (down_25 if m.group(2) == "26" else down_41)[int(m.group(3))])
            n_down += 1
            continue
        m = re.match(r"t (layers\.\d+\.attn\.compressor\.(?:wkv|wgate))\.weight (\w+) (\d+) (\d+) (\d+) ([0-9 ]+)$", ln)
        if m:
            name, dt, fi, off, nb, shape = m.groups()
            dims = [int(x) for x in shape.split()]
            if dims[0] == 1024:   # ratio 4: [2*head, dim] -> [head, dim]
                new.append("t %s.weight %s %s %s %d %s" %
                           (name, dt, fi, off, int(nb) // 2, " ".join(map(str, [512] + dims[1:]))))
                n_cut += 1
                continue
        new.append(ln)
    p.write_text("\n".join(new) + "\n")
    print("%s: %d down matrices replaced, %d compressor matrices cut (test only)" % (dst, n_down, n_cut))
    return 0


if __name__ == "__main__":
    sys.exit(main())
