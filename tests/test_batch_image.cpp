// Consistency test for the batch image inference API (official
// set_image_batch flow): sam3_encode_image_batch + sam3_segment_pcs_batch
// must produce the same detections as encoding and querying each image
// separately — the detector attends within each image only, so per-image
// forwards are numerically identical to the official batched forward.
//
// Usage: test_batch_image <model.gguf> <imageA> [imageB]
#include "sam3.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while (0)

static std::string fnv(const sam3_detection& d) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, size_t n) {
        const auto* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    mix(&d.instance_id, sizeof(d.instance_id));
    mix(&d.score, sizeof(d.score));
    mix(&d.box, sizeof(d.box));
    uint32_t w = d.mask.width, ht = d.mask.height;
    mix(&w, sizeof(w)); mix(&ht, sizeof(ht));
    if (!d.mask.data.empty()) mix(d.mask.data.data(), d.mask.data.size());
    char out[17];
    snprintf(out, sizeof(out), "%016llx", (unsigned long long)h);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <imageA> [imageB]\n", argv[0]);
        return 1;
    }
    sam3_params p;
    p.model_path = argv[1];
    p.n_threads  = 8;
    auto model = sam3_load_model(p);
    CHECK(model != nullptr);
    if (!model) return 1;
    if (sam3_is_visual_only(*model)) {
        fprintf(stderr, "SKIP: visual-only model has no detector path\n");
        return 0;
    }

    sam3_image imgs[2] = {sam3_load_image(argv[2]), {}};
    const int n = argc > 3 ? 2 : 1;
    if (n == 2) imgs[1] = sam3_load_image(argv[3]);
    for (int i = 0; i < n; ++i) CHECK(!imgs[i].data.empty());

    sam3_pcs_params pcs;
    pcs.text_prompt = "animal";
    pcs.score_threshold = 0.5f;

    // ── Per-image reference: encode + PCS separately ──────────────────────
    std::vector<sam3_result> ref(n);
    for (int i = 0; i < n; ++i) {
        auto st = sam3_create_state(*model, p);
        CHECK(st != nullptr);
        CHECK(sam3_encode_image(*st, *model, imgs[i]));
        ref[i] = sam3_segment_pcs(*st, *model, pcs);
        sam3_free_state(*st);
    }

    // ── Batch: encode both, then query the batch ──────────────────────────
    auto st = sam3_create_state(*model, p);
    CHECK(st != nullptr);
    CHECK(sam3_encode_image_batch(*st, *model, imgs, n) == n);
    std::vector<sam3_result> got(n);
    CHECK(sam3_segment_pcs_batch(*st, *model, pcs, got.data(), n) == n);

    // ── Compare ────────────────────────────────────────────────────────────
    for (int i = 0; i < n; ++i) {
        fprintf(stderr, "image %d: ref %zu det(s), batch %zu det(s)\n",
                i, ref[i].detections.size(), got[i].detections.size());
        CHECK(ref[i].detections.size() == got[i].detections.size());
        const size_t k = std::min(ref[i].detections.size(), got[i].detections.size());
        for (size_t d = 0; d < k; ++d) {
            const std::string hr = fnv(ref[i].detections[d]);
            const std::string hb = fnv(got[i].detections[d]);
            CHECK(hr == hb);
            if (hr != hb)
                fprintf(stderr, "  det %zu hash mismatch: ref %s vs batch %s\n",
                        d, hr.c_str(), hb.c_str());
        }
    }

    // ── After the batch, the live state must still work per-image ─────────
    CHECK(sam3_encode_image(*st, *model, imgs[0]));
    sam3_result live = sam3_segment_pcs(*st, *model, pcs);
    CHECK(live.detections.size() == ref[0].detections.size());

    if (g_fail == 0) {
        fprintf(stderr, "PASS: batch inference matches per-image inference\n");
        sam3_free_state(*st);
        return 0;
    }
    fprintf(stderr, "%d check(s) failed\n", g_fail);
    sam3_free_state(*st);
    return 1;
}
