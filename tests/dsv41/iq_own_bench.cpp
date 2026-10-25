// Strata. The DeepSeek-V4.1 decode lane's i-quant CPU rows (src/dsv41/ds4_iq_cpu.cpp: Maya's AVX2 / AVX-512 rows for
// IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S) against this build's native lane (ds4_gg_cpu.cpp: ggml-cpu's vec_dot).  No GPU and
// no model: random blocks of the UD-IQ2_XXS pack's expert types (gate/up IQ2_XXS, down IQ3_XXS) by default.
//
//   ds4_iq_own_bench [--mode auto|own|native] [--tg 16] [--td 18] [--e 4096] [--ff 2048] [--nt 1] [--experts 48]
//                    [--threads 12] [--reps 3]
//
// What it prints, in order:
//   - the row choice ds4_iq_cpu.cpp's own() made here for each type (its own rows, or the native lane), and with
//     DS4_IQ_VERBOSE=1 the row-level timings behind it (native vs ours, and the second form: vpgatherdd / one table);
//   - the expert pipeline the engine's CPU lane runs per token - gate/up rows, quantize h, down rows - over N experts
//     on T threads: the native lane (ggml's vec_dot), the engine's lane (iq_cpu_*, i.e. the rows own() chose), GB/s of
//     weights read, and the speed-up between them;
//   - whether the two paths' outputs are the same bits (they are meant to be: the rows are ggml's arithmetic).
// The numbers are this CPU's: a Zen 2 has AVX2 only, so the 512-bit rows are reported as not run here; --mode own with
// DS4_IQ_AVX2=1 / DS4_IQ_512=0 compares the AVX2 forms on a newer CPU.
#include "ds4_gguf_cpu.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds4::gguf;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

// random bytes with a plausible F16 scale at each block's start (IQ1_M keeps its f16 in the scale words, as in
// tests/dsv41/gg_dot_bench.cpp; none of the four types here does)
static void fill_blocks(uint8_t* w, size_t bytes, int type, uint64_t seed) {
    uint64_t st = 0x9E3779B97F4A7C15ull ^ (seed * 0xD1B54A32D192ED03ull);
    auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    for (size_t o = 0; o + 8 <= bytes; o += 8) { const uint64_t v = next(); std::memcpy(w + o, &v, 8); }
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bb <= bytes; o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        std::memcpy(w + o, &d, sizeof d);
    }
}

int main(int argc, char** argv) {
    int tg = 16, td = 18, E = 4096, FF = 2048, nt = 1, threads = 12, ne = 48, reps = 3;
    std::string mode = "auto";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&] { return i + 1 < argc ? std::atol(argv[++i]) : 0L; };
        if (a == "--mode") mode = i + 1 < argc ? argv[++i] : mode;
        else if (a == "--tg") tg = (int) val();
        else if (a == "--td") td = (int) val();
        else if (a == "--e") E = (int) val();
        else if (a == "--ff") FF = (int) val();
        else if (a == "--nt") nt = std::max(1, (int) val());
        else if (a == "--experts") ne = std::max(1, (int) val());
        else if (a == "--threads") threads = std::max(1, (int) val());
        else if (a == "--reps") reps = std::max(1, (int) val());
    }
    if (mode == "own") setenv("DS4_IQ_OWN", "1", 1);          // (before the first own() call: it decides once)
    else if (mode == "native") setenv("DS4_IQ_OWN", "0", 1);

    IqCpu f;
    std::string err;
    if (!iq_cpu_fmt(tg, td, E, FF, f, err)) {
        std::fprintf(stderr, "iq_cpu_fmt(%d, %d, %d, %d): %s\n", tg, td, E, FF, err.c_str());
        return 2;
    }
    // (no gg_cpu_set_iq256: the reference is the engine's default lane - ggml-cpu's dot at one token, this build's own
    // multi-token rows from two, ds4_gg_cpu.cpp - which is the lane iq_cpu_* falls back to)

    const size_t gbytes = f.gu_row * (size_t) FF, dbytes = f.d_row * (size_t) E, ebytes = 2 * gbytes + dbytes;
    std::vector<uint8_t> gw((size_t) ne * gbytes), uw((size_t) ne * gbytes), dw((size_t) ne * dbytes);
    for (int e = 0; e < ne; ++e) {
        fill_blocks(&gw[(size_t) e * gbytes], gbytes, tg, 3 * (uint64_t) e + 1);
        fill_blocks(&uw[(size_t) e * gbytes], gbytes, tg, 3 * (uint64_t) e + 2);
        fill_blocks(&dw[(size_t) e * dbytes], dbytes, td, 3 * (uint64_t) e + 3);
    }
    uint64_t st = 12345;
    auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    std::vector<std::vector<uint8_t>> act((size_t) nt, std::vector<uint8_t>(f.act_bytes));
    for (int t = 0; t < nt; ++t) {
        std::vector<float> x((size_t) E);
        for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
        iq_cpu_quant(f.gu_act, x.data(), E, act[(size_t) t].data());
    }
    std::vector<float> h((size_t) ne * nt * FF), y((size_t) ne * nt * E);
    std::vector<float> h0, y0;
    std::vector<uint8_t> hq((size_t) ne * nt * f.h_bytes);
    const int RB = 64;
    auto run_jobs = [&](long njobs, const std::function<void(long)>& fn) {
        std::atomic<long> cur{0};
        std::vector<std::thread> th;
        for (int k = 0; k < threads; ++k)
            th.emplace_back([&] { for (;;) { const long j = cur.fetch_add(1); if (j >= njobs) break; fn(j); } });
        for (auto& t : th) t.join();
    };
    // native: gg_cpu_* (ggml's vec_dot) straight; engine: iq_cpu_* (the rows own() chose, the lane the engine runs)
    auto pass = [&](bool engine) {
        double best = 1e30;
        for (int r = 0; r < reps + 1; ++r) {
            const auto t0 = clk::now();
            const long gb = (FF + RB - 1) / RB, db = (E + RB - 1) / RB;
            run_jobs((long) ne * gb, [&](long j) {
                const int e = (int) (j / gb), r0 = (int) (j % gb) * RB, r1 = std::min(FF, r0 + RB);
                const void* ap[64];
                float* hp[64];
                for (int t = 0; t < nt; ++t) { ap[t] = act[(size_t) t].data(); hp[t] = &h[((size_t) e * nt + t) * FF]; }
                if (engine) iq_cpu_gate_up(f, &gw[(size_t) e * gbytes], &uw[(size_t) e * gbytes], ap, nt, hp, r0, r1, 10.0f);
                else gg_cpu_gate_up(f, &gw[(size_t) e * gbytes], &uw[(size_t) e * gbytes], ap, nt, hp, r0, r1, 10.0f);
            });
            run_jobs((long) ne * nt, [&](long j) { iq_cpu_quant(f.d_act, &h[(size_t) j * FF], FF, &hq[(size_t) j * f.h_bytes]); });
            run_jobs((long) ne * db, [&](long j) {
                const int e = (int) (j / db), r0 = (int) (j % db) * RB, r1 = std::min(E, r0 + RB);
                const void* hp[64];
                float* yp[64];
                for (int t = 0; t < nt; ++t) { hp[t] = &hq[((size_t) e * nt + t) * f.h_bytes]; yp[t] = &y[((size_t) e * nt + t) * E]; }
                if (engine) iq_cpu_down(f, &dw[(size_t) e * dbytes], hp, nt, yp, r0, r1);
                else gg_cpu_down(f, &dw[(size_t) e * dbytes], hp, nt, yp, r0, r1);
            });
            if (r > 0) best = std::min(best, secs(t0));
        }
        return best;
    };

    const double gb = (double) ne * ebytes / 1e9;
    std::printf("gate/up %s (%zu B a row) down %s (%zu B a row), %d experts = %.2f GB, n_embd %d n_ff %d, %d token(s)\n",
                ggml_type_name((ggml_type) tg), f.gu_row, ggml_type_name((ggml_type) td), f.d_row, ne, gb, E, FF, nt);
    const bool own_gu = iq_cpu_own_rows(tg), own_d = iq_cpu_own_rows(td);
    std::printf("own(): gate/up rows 1 token %s, 4 tokens %s; down rows 1 token %s, 4 tokens %s\n",
                own_gu ? "ours" : "engine", iq_cpu_own_rows(tg, 4) ? "ours" : "engine", own_d ? "ours" : "engine",
                iq_cpu_own_rows(td, 4) ? "ours" : "engine");

    const double t_gg = pass(false);
    y0 = y;
    const double t_en = pass(true);
    const bool same = std::memcmp(y0.data(), y.data(), y.size() * 4) == 0;
    // the bit-exact test on the whole pipeline: every down-row output of the two lanes, compared as raw float bits
    std::printf("bit check: %zu down-row outputs (%d experts x %d token(s) x %d rows), engine default vs the rows: %s\n",
                y.size(), ne, nt, E, same ? "identical" : "DIFFERENT");
    std::printf("engine default %.3f s = %.1f GB/s (%.3f ms an expert) | with the rows %.3f s = %.1f GB/s (%.3f ms an "
                "expert) | x%.2f %s\n", t_gg, gb / t_gg, t_gg / ne * 1e3, t_en, gb / t_en, t_en / ne * 1e3, t_gg / t_en,
                same ? "bit-identical" : "DIFFERENT BITS");
    return same ? 0 : 1;
}
