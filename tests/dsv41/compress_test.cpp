// Strata. The DeepSeek compressors (ds4_kernels.cu compress_win) against a double reference written from llama.cpp's
// graphs - build_hca_compressed_kv_from_state (ratio 128) and build_overlap_compressed_kv_from_state (ratio 4, the
// overlap): per value, a softmax over the rows a group reads, then the weighted sum, rmsnorm(w).  The four shapes the
// engine runs: deepseek4's CSA (coff 2: a row's first head values are the NEXT group's previous half, its last head the
// group's own half, so a group mixes the previous group's first halves with its own second halves) with the position
// table, V4.1's ratio 4 (coff 1) and ratio 128 (HCA), and CSA without a table (the first group of a sequence, whose
// previous group does not exist, is covered by the run from position 0).  Chunks of arbitrary length at arbitrary
// positions: the ring must carry a group across a call.
//   ds4_compress_test [--verbose]
#include "../../src/dsv41/ds4_kernels.cuh"

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ds4 = strata::ds4;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

namespace {

// llama.cpp's compressor, in double: the state ring a row a token (ape[slot % ratio] added as the row is written), a
// completed group pooling the rows it reads.
struct Ref {
    int ratio = 0, coff = 1, slots = 0, d = 0, wide = 0;
    float eps = 1e-6f;
    std::vector<double> ape;              // ratio * wide (empty: none)
    std::vector<double> ring_kv, ring_sc;  // slots * wide
    std::vector<std::vector<double>> groups;

    Ref(int ratio_, int coff_, int d_, const std::vector<float>& ape_)
        : ratio(ratio_), coff(coff_), d(d_), wide(coff_ * d_), ape(ape_.begin(), ape_.end()) {
        slots = coff * ratio;
        ring_kv.assign((size_t) slots * wide, 0.0);
        ring_sc.assign((size_t) slots * wide, 0.0);
    }

    void run(int p0, const std::vector<float>& kv, const std::vector<float>& sc, const std::vector<float>& w) {
        const int T = (int) (kv.size() / (size_t) wide);
        for (int t = 0; t < T; ++t) {
            const int pos = p0 + t, slot = pos % slots;
            for (int e = 0; e < wide; ++e) {
                ring_kv[(size_t) slot * wide + e] = kv[(size_t) t * wide + e];
                double v = sc[(size_t) t * wide + e];
                if (!ape.empty()) v += ape[(size_t) (slot % ratio) * wide + e];
                ring_sc[(size_t) slot * wide + e] = v;
            }
            if ((pos + 1) % ratio != 0) continue;
            const int b0 = pos + 1 - ratio;
            const bool prev = coff > 1 && b0 >= ratio;
            std::vector<double> pool((size_t) d, 0.0);
            for (int e = 0; e < d; ++e) {
                std::vector<double> s, v;
                const auto add = [&](int pos2, int e2) {
                    const int s2 = pos2 % slots;
                    s.push_back(ring_sc[(size_t) s2 * wide + e2]);
                    v.push_back(ring_kv[(size_t) s2 * wide + e2]);
                };
                if (prev)
                    for (int i = 0; i < ratio; ++i) add(b0 - ratio + i, e);
                for (int i = 0; i < ratio; ++i) add(b0 + i, coff > 1 ? d + e : e);
                const double m = *std::max_element(s.begin(), s.end());
                double den = 0.0, acc = 0.0;
                for (size_t i = 0; i < s.size(); ++i) {
                    const double p = std::exp(s[i] - m);
                    den += p;
                    acc += p * v[i];
                }
                pool[(size_t) e] = acc / den;
            }
            double ss = 0.0;
            for (int e = 0; e < d; ++e) ss += pool[(size_t) e] * pool[(size_t) e];
            const double inv = 1.0 / std::sqrt(ss / (double) d + (double) eps);
            std::vector<double> o((size_t) d);
            for (int e = 0; e < d; ++e) o[(size_t) e] = pool[(size_t) e] * inv * (double) w[(size_t) e];
            groups.push_back(std::move(o));
        }
    }
};

struct F {
    std::mt19937 rng{1234};
    std::uniform_real_distribution<float> u{-1.0f, 1.0f};
    float v() { return u(rng); }
};

template <typename T> T* up(const std::vector<T>& x) {
    T* p = nullptr;
    CK(hipMalloc(&p, std::max<size_t>(1, x.size() * sizeof(T))));
    if (!x.empty()) CK(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

// One run: T tokens at p0, chunks of `step` through the kernel, compared group by group with the reference.
int once(const char* name, int ratio, int coff, int d, int n_tok, int step, bool with_ape, bool verbose) {
    F f;
    const int wide = coff * d, slots = coff * ratio;
    std::vector<float> w((size_t) d), ape(with_ape ? (size_t) ratio * wide : 0);
    for (float& x : w) x = 0.5f + 0.5f * std::fabs(f.v());
    for (float& x : ape) x = 0.1f * f.v();
    std::vector<float> kv((size_t) n_tok * wide), sc((size_t) n_tok * wide);
    for (float& x : kv) x = f.v();
    for (float& x : sc) x = f.v();

    Ref ref(ratio, coff, d, ape);
    std::vector<float> st_kv((size_t) slots * wide, 0.0f), st_sc((size_t) slots * wide, 0.0f);
    float *d_kv = up(st_kv), *d_sc = up(st_sc);
    float* d_ape = with_ape ? up(ape) : nullptr;
    float* d_w = up(w);
    float* d_lat = up(std::vector<float>((size_t) n_tok * d + d));
    for (int p0 = 0; p0 < n_tok; p0 += step) {
        const int T = std::min(step, n_tok - p0);
        std::vector<float> kvr(kv.begin() + (size_t) p0 * wide, kv.begin() + (size_t) (p0 + T) * wide);
        std::vector<float> scr(sc.begin() + (size_t) p0 * wide, sc.begin() + (size_t) (p0 + T) * wide);
        float* d_kv_in = up(kvr);
        float* d_sc_in = up(scr);
        ds4::compress_win(d_kv_in, d_sc_in, d_kv, d_sc, ratio, coff, slots, p0, T, d, d_w, 1e-6f, d_ape, d_lat,
                          nullptr);
        CK(hipDeviceSynchronize());
        std::vector<float> got((size_t) T * d / ratio + d);
        CK(hipMemcpy(got.data(), d_lat, got.size() * sizeof(float), hipMemcpyDeviceToHost));
        const int ng = (p0 + T) / ratio - p0 / ratio;   // groups this call completed
        CK(hipFree(d_kv_in));
        CK(hipFree(d_sc_in));
        const size_t before = ref.groups.size();
        ref.run(p0, kvr, scr, w);
        const size_t added = ref.groups.size() - before;
        if ((int) added != ng) {
            std::printf("%s: %d groups completed, the reference had %zu (positions %d + %d)\n", name, ng, added, p0, T);
            return 1;
        }
        double worst = 0.0;
        int wi = 0, wj = 0;
        for (size_t g = 0; g < added; ++g)
            for (int e = 0; e < d; ++e) {
                const double a = got[g * (size_t) d + e], b = ref.groups[before + g][(size_t) e];
                const double diff = std::fabs(a - b) / (1.0 + std::fabs(b));
                if (diff > worst) worst = diff, wi = (int) g, wj = e;
            }
        if (worst > 2e-5) {
            std::printf("%s: positions %d + %d: group %d value %d is %.8g, the reference %.8g (relative %.2g)\n", name,
                        p0, T, wi, wj, (double) got[(size_t) wi * d + wj], ref.groups[before + (size_t) wi][(size_t) wj],
                        worst);
            return 1;
        }
        if (verbose)
            std::printf("%s: positions %3d + %-3d %2zu groups, worst %.2g\n", name, p0, T, added, worst);
    }
    CK(hipFree(d_kv));
    CK(hipFree(d_sc));
    CK(hipFree(d_w));
    CK(hipFree(d_lat));
    if (d_ape != nullptr) CK(hipFree(d_ape));
    std::printf("%s: %d positions, chunks of %d, %s the table: %zu groups match the reference\n", name, n_tok, step,
                with_ape ? "with" : "without", ref.groups.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool verbose = argc > 1 && std::string(argv[1]) == "--verbose";
    int bad = 0;
    bad += once("CSA (ratio 4, two halves), chunks of 5", 4, 2, 512, 37, 5, true, verbose);
    bad += once("CSA, chunks of 1 (a token a call)", 4, 2, 64, 23, 1, true, verbose);
    bad += once("CSA, chunks of 16 (several groups a call)", 4, 2, 64, 40, 16, true, verbose);
    bad += once("CSA without the table", 4, 2, 64, 21, 4, false, verbose);
    bad += once("V4.1 (ratio 4, one half)", 4, 1, 64, 29, 7, false, verbose);
    bad += once("HCA (ratio 128)", 128, 1, 64, 300, 64, true, verbose);
    bad += once("HCA, chunks of 17 (a group across calls)", 128, 1, 64, 260, 17, true, verbose);
    std::printf("%s\n", bad == 0 ? "all compressors match the reference" : "FAILED");
    return bad == 0 ? 0 : 1;
}
