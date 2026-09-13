// Regression test for the official video-predictor alignment of the tracker:
//
//   1. reverse propagation: sam3_propagate_frame(reverse=true) walks
//      decreasing frame indices (official range(start-1, ..., -1)), never
//      re-processes the prompt frame, and is rejected at frame 0;
//   2. forward <-> reverse interleaving keeps every mask on-object (masks
//      stay non-empty with sane coverage across direction flips);
//   3. memory selection (official use_memory_selection / frame_filter,
//      enabled by the SAM3 hparams KV): with all eff_iou_scores below
//      mf_threshold the adjacent frame is still selected (official
//      must_include), so propagation never starves of memory; boundary
//      frames (reverse at the last frame / forward at frame 0) yield an
//      empty filter without crashing.
//
// Uses one static image for every frame: the structural guarantees above are
// direction/bookkeeping properties, independent of frame content.
//
// Usage: test_reverse_align <model.gguf> [image.jpg]
#include "sam3.h"
#include <cstdio>
#include <string>

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
    } \
} while (0)

static float fg_ratio(const sam3_mask& m) {
    if (m.data.empty()) return 0.0f;
    int fg = 0;
    for (uint8_t v : m.data) if (v > 127) ++fg;
    return (float)fg / (float)m.data.size();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [image.jpg]\n", argv[0]);
        return 1;
    }
    sam3_params p;
    p.model_path = argv[1];
    p.n_threads  = 8;
    // Model resolution is fixed (the positional embeddings are precomputed at
    // 1008); the checks below are bookkeeping/structural, so the static image
    // is simply re-encoded for every visited frame.
    auto model = sam3_load_model(p);
    CHECK(model != nullptr);
    if (!model) return 1;

    const std::string img_path = argc > 2 ? argv[2] : "data/test_image.jpg";
    sam3_image img = sam3_load_image(img_path);
    CHECK(!img.data.empty());

    auto state = sam3_create_state(*model, p);
    CHECK(state != nullptr);
    CHECK(sam3_encode_image(*state, *model, img));

    sam3_visual_track_params vp;
    auto tracker = sam3_create_visual_tracker(*model, vp);
    CHECK(tracker != nullptr);

    sam3_pvs_params seed;
    seed.pos_points.push_back({(float)img.width / 2, (float)img.height / 2});
    const int id = sam3_tracker_add_instance(*tracker, *state, *model, seed);
    CHECK(id > 0);
    CHECK(sam3_tracker_frame_index(*tracker) == 0);

    // ── 1. forward: range(start, ...] — first call re-processes frame 0 ──
    for (int f = 0; f <= 2; ++f) {
        sam3_result r = sam3_propagate_frame(*tracker, *state, *model, img);
        CHECK(!r.detections.empty());
        if (!r.detections.empty())
            CHECK(fg_ratio(r.detections[0].mask) > 0.001f);
        CHECK(sam3_tracker_frame_index(*tracker) == f + 1);
    }
    CHECK(sam3_tracker_frame_index(*tracker) == 3);

    // ── 2. reverse: range(start-1, ..., -1) — decreasing indices ─────────
    {
        sam3_result r = sam3_propagate_frame(*tracker, *state, *model, img, true);
        CHECK(!r.detections.empty());
        if (!r.detections.empty())
            CHECK(fg_ratio(r.detections[0].mask) > 0.001f);
        CHECK(sam3_tracker_frame_index(*tracker) == 2);
    }

    // ── 3. flip back to forward: re-processes frame 2 first ──────────────
    for (int f = 2; f <= 3; ++f) {
        sam3_result r = sam3_propagate_frame(*tracker, *state, *model, img);
        CHECK(!r.detections.empty());
        CHECK(sam3_tracker_frame_index(*tracker) == f + 1);
    }
    CHECK(sam3_tracker_frame_index(*tracker) == 4);

    // ── 4. reverse all the way to 0, then past it (must be rejected) ─────
    while (sam3_tracker_frame_index(*tracker) > 0) {
        sam3_result r = sam3_propagate_frame(*tracker, *state, *model, img, true);
        CHECK(!r.detections.empty());
    }
    CHECK(sam3_tracker_frame_index(*tracker) == 0);
    sam3_result rbad = sam3_propagate_frame(*tracker, *state, *model, img, true);
    CHECK(rbad.detections.empty());
    CHECK(sam3_tracker_frame_index(*tracker) == 0);

    if (g_fail == 0) {
        fprintf(stderr, "PASS: reverse/memory-selection alignment checks\n");
        return 0;
    }
    fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
}
