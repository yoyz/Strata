#!/bin/bash
# strata-dsv41 (the DeepSeek engine) as its own feed package for Linux AMD: the `dsv41` member of the update feed, whose
# entrypoint is strata-dsv41 itself (strata_control/dsv41_engine.py installs it in <data>/engine-dsv41/).  The V4.1
# packages are made this way; the Qwen engine is not part of them and its package is untouched.
# Run on a Linux box that has ROCm (TheRock Python wheels are enough).  All settings are environment variables:
#
#   VARIANT=rx9070     rx9070 (gfx1200 gfx1201) | rx7900 (gfx1100 gfx1101 gfx1102) | rx6000 (gfx1030 gfx1031) | halo (gfx1151) | igpu (gfx1103 gfx1150) | a name + ARCHS=
#   ARCHS="gfx1200;gfx1201"          the code objects (default: the variant's)
#   SRC=<this checkout>              the Strata source (default: the checkout this script is in)
#   B=<build dir>                    where the engine is built (default: $OUT/bd)
#   LLAMA=<llama.cpp checkout>       at the pinned commit (setup.py LLAMA_CPP_COMMIT)
#   HP=<ROCm devel root>  CORE=<ROCm core root>  LIBW=<ROCm libraries root>   (TheRock: _rocm_sdk_devel, _rocm_sdk_core, _rocm_sdk_libraries_*)
#   OUT=<work folder>  PKGVER=0.1.50  COMMIT=<source commit for the build id>  JOBS=16
#   KEY=<feed-key.json>              signs the package's manifest (packaging/feed_tool.py); without it the package is made
#                                    unsigned and the entry is not installable by a PC that trusts the release key
#   NO_PACKAGE=1                     stop after the stage folder (the package's future contents)
#
# Result: $OUT/components-dsv41/<version>-<platform>-hip-<variant>/ (the component folder, signed when KEY is given) and
# $OUT/dsv41/strata-dsv41-<folder>.<fmt> + .entry.json (what `feed_tool.py build-feed --dsv41 $OUT/dsv41` takes).
set -u
V=${VARIANT:?VARIANT=rx9070|rx7900|rx6000|halo|igpu|<name> with ARCHS=}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:?LLAMA=<llama.cpp checkout at the pinned commit>}
HP=${HP:?HP=<ROCm devel root>}
CORE=${CORE:-$HP}
LIBW=${LIBW:-$HP}
OUT=${OUT:-$PWD/work-dsv41-$V}
B=${B:-$OUT/bd}
PKGVER=${PKGVER:-0.1.50}
COMMIT=${COMMIT:-$(git -C "$SRC" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)}
JOBS=${JOBS:-16}
FMT=${FMT:-tar.gz}
case "$V" in
  rx9070) ARCHS=${ARCHS:-gfx1200;gfx1201} ;;
  rx7900) ARCHS=${ARCHS:-gfx1100;gfx1101;gfx1102} ;;
  rx6000) ARCHS=${ARCHS:-gfx1030;gfx1031} ;;
  halo)   ARCHS=${ARCHS:-gfx1151} ;;
  igpu)   ARCHS=${ARCHS:-gfx1103;gfx1150} ;;
  *)      ARCHS=${ARCHS:?ARCHS=gfx...} ;;
esac
ARCHLIST=${ARCHS//;/ }
DATAARCH=${ARCHS//;/,}
mkdir -p "$OUT"; S=$OUT/build.status
st() { echo "$* $(date +%T)" >> "$S"; }
: > "$S"
command -v patchelf >/dev/null || { echo "patchelf is needed (the package's libraries must search \$ORIGIN, not the build machine's ROCm): install it and run again" >&2; exit 1; }
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
HIPDEFS=(-DCMAKE_HIP_COMPILER=$HP/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$HP "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS"
         "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode")
st "start $V ($ARCHS) commit $COMMIT"
cd "$SRC" || exit 1
# the release shape: portable CPU kernels (AVX2, no AVX-512), the CPU lane's native expert kernels on, RUNPATH $ORIGIN
cmake -G Ninja -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON "-DCMAKE_HIP_ARCHITECTURES=$ARCHS" "${HIPDEFS[@]}" \
  -DSTRATA_GGML_DIR="$LLAMA" -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" > "$OUT/cfg.log" 2>&1
st "configured rc=$?"
nice -n 10 cmake --build "$B" --target strata-dsv41 -j "$JOBS" > "$OUT/build.log" 2>&1
st "built rc=$?"
[ -x "$B/strata-dsv41" ] || { st "ABORT no strata-dsv41"; tail -20 "$OUT/build.log" >> "$S"; exit 1; }
VERSION=$(LD_LIBRARY_PATH="$LIBW/lib:$HP/lib:$CORE/lib${SYSDEPS:+:$HP/lib/rocm_sysdeps/lib}" "$B/strata-dsv41" --version 2>/dev/null | awk '{print $2}')
[ "$VERSION" = "$PKGVER" ] || { st "ABORT built $VERSION, packaging $PKGVER"; exit 1; }   # the folder name is the number the program prints
STAGE=$OUT/stage-dsv41; rm -rf "$STAGE"; mkdir -p "$STAGE"
ARCHJSON=$(for a in $ARCHLIST; do printf '"%s", ' "$a"; done | sed 's/, $//')
cat > "$STAGE/BUILD.json" <<EOJ
{"version": "$PKGVER", "source": "f-dsv41-hip $COMMIT", "variant": "$V", "archs": [$ARCHJSON], "portable": true, "dsv41": true}
EOJ
cp "$B/strata-dsv41" "$STAGE/"
DATA=(--data "$LIBW/lib/rocblas/library=rocblas/library" --data "$LIBW/lib/hipblaslt/library=hipblaslt/library" --data-arch "$DATAARCH")
python3 packaging/collect_libs.py --binary "$STAGE/strata-dsv41" --search "$LIBW/lib" --search "$HP/lib" --search "$CORE/lib" \
  --search "$CORE/lib/llvm/lib" ${SYSDEPS:+--search "$HP/lib/rocm_sysdeps/lib"} \
  --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' --out "$STAGE" "${DATA[@]}" >> "$OUT/collect.log" 2>&1
st "collected rc=$?"
for f in "$STAGE"/*.so "$STAGE"/*.so.*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"; done
st "RUNPATH: $(readelf -d "$STAGE/strata-dsv41" | grep -E 'RPATH|RUNPATH' | cut -d'[' -f2 | tr -d ']' | paste -sd' ')"
python3 packaging/stage_tuning.py --stage "$STAGE" --archs "$ARCHLIST" --rocm "$HP" >> "$OUT/stage_tuning.log" 2>&1
st "tuning tables rc=$?: $(tail -n 1 "$OUT/stage_tuning.log")"
[ -n "${NO_PACKAGE:-}" ] && { st "STAGE-DONE"; exit 0; }
rm -rf "$OUT/components-dsv41"; mkdir -p "$OUT/components-dsv41"
ARCHARGS=""; for a in $ARCHLIST; do ARCHARGS="$ARCHARGS --arch $a"; done
python3 packaging/package_engine.py --src "$STAGE" --backend hip --version "$PKGVER" --out "$OUT/components-dsv41" \
  --entrypoint strata-dsv41 $ARCHARGS --variant "$V" --build-id "$COMMIT+rocm" --tree \
  --notes "DeepSeek engine, AMD $V build (strata-dsv41: DeepSeek-V4.1 and DeepSeek-V4-Flash-0731)" > "$OUT/package.log" 2>&1
st "packaged rc=$?"
cat "$OUT/package.log" >> "$S"
PKG=$(find "$OUT/components-dsv41" -mindepth 1 -maxdepth 1 -type d -name "$PKGVER-*" | head -1)
[ -n "$PKG" ] || { st "ABORT no component folder"; exit 1; }
if [ -n "${KEY:-}" ]; then
  python3 packaging/feed_tool.py sign-package "$PKG" --key "$KEY" >> "$OUT/sign.log" 2>&1
  st "signed rc=$?"
else
  echo "note: no KEY given: the package is unsigned and a PC trusting the release key will refuse it" >> "$S"
fi
rm -rf "$OUT/dsv41"; mkdir -p "$OUT/dsv41"
python3 packaging/feed_tool.py pack-engine "$PKG" --out "$OUT/dsv41" --format "$FMT" --prefix strata-dsv41 --url-prefix "" >> "$OUT/pack.log" 2>&1
st "packed rc=$?"
cat "$OUT/pack.log" >> "$S"
st "BUILD-DONE"
