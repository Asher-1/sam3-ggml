// A/B forward-propagation probe for old/new library comparison.
//
// Compiled once against the OLD library (build/libsam3.so, pre-alignment
// code) and once against the NEW one (build-vis/libsam3.so); stdout is a
// deterministic per-frame report (detection count, score/box at 6 decimals,
// mask FNV hash, per-phase ms) so a plain diff shows numerical drift and the
// timings show the speed delta.
//
//   ./ab_forward_probe <model.gguf> <image> [n_frames=8]
#include "sam3.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static std::string fnv(const sam3_detection& d) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, size_t n) {
        const auto* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    mix(&d.instance_id, sizeof(d.instance_id));
    mix(&d.score, sizeof(d.score));
    mix(&d.iou_score, sizeof(d.iou_score));
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
        fprintf(stderr, "Usage: %s <model.gguf> <image> [n_frames]\n", argv[0]);
        return 1;
    }
    const int n_frames = argc > 3 ? atoi(argv[3]) : 8;

    sam3_params p;
    p.model_path = argv[1];
    p.n_threads  = 8;
    p.use_gpu    = false;

    auto model = sam3_load_model(p);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    sam3_image img = sam3_load_image(argv[2]);
    if (img.data.empty()) { fprintf(stderr, "image failed\n"); return 1; }

    auto state = sam3_create_state(*model, p);
    double t0 = now_ms();
    if (!sam3_encode_image(*state, *model, img)) { fprintf(stderr, "encode failed\n"); return 1; }
    printf("ENCODE %.3f ms\n", now_ms() - t0);

    sam3_visual_track_params vp;
    auto tracker = sam3_create_visual_tracker(*model, vp);
    sam3_pvs_params seed;
    seed.pos_points.push_back({(float)img.width / 2, (float)img.height / 2});
    const int id = sam3_tracker_add_instance(*tracker, *state, *model, seed);
    printf("ADD id=%d frame_index=%d\n", id, sam3_tracker_frame_index(*tracker));

    double prop_sum = 0.0;
    for (int f = 0; f < n_frames; ++f) {
        t0 = now_ms();
        sam3_result r = sam3_propagate_frame(*tracker, *state, *model, img);
        double dt = now_ms() - t0;
        prop_sum += dt;
        printf("FRAME %d %.3f ms dets=%zu", f, dt, r.detections.size());
        for (size_t d = 0; d < r.detections.size(); ++d) {
            const auto& det = r.detections[d];
            int fg = 0;
            for (uint8_t v : det.mask.data) if (v > 127) ++fg;
            printf(" [id=%d score=%.6f iou=%.6f box=(%.2f,%.2f,%.2f,%.2f) fg=%d hash=%s]",
                   det.instance_id, det.score, det.iou_score,
                   det.box.x0, det.box.y0, det.box.x1, det.box.y1,
                   fg, fnv(det).c_str());
        }
        printf("\n");
    }
    printf("SUMMARY propagate_total=%.3f ms avg=%.3f ms frames=%d final_frame_index=%d\n",
           prop_sum, prop_sum / n_frames, n_frames, sam3_tracker_frame_index(*tracker));
    return 0;
}
