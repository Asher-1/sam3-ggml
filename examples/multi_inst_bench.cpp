/**
 * sam3_multi_bench — multi-instance tracking benchmark: SAM 3.1 (mux bucket
 * pass) vs SAM 3.0 (per-instance propagation) at N simultaneous instances.
 *
 * Seeds N instances with point prompts on frame 0 (fixed grid over distinct
 * market objects), then tracks frames 1..K-1.  Reports seed time, average
 * per-frame track time and GPU memory (device used = total - free, sampled
 * after the last track frame; assumes a single-tenant GPU).
 *
 * Usage:
 *   sam3_multi_bench --model <path.gguf> [--instances 1,4,8,16]
 *                    [--n-frames 4] [--video <path>] [--threads 4] [--cpu]
 */

#include "sam3.h"
#include "example_debug.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 16 distinct object centers on the market scene (960x540, frame 0).
static const float kPoints[][2] = {
    {315, 250}, {700,  60}, {870, 250}, {820, 130},
    {700, 130}, {430, 130}, {250, 100}, { 90, 200},
    {640, 300}, {560, 330}, {760, 350}, {540,  70},
    {150, 380}, {380, 420}, {700, 430}, {900,  60},
};
static const int kNumPoints = (int)(sizeof(kPoints) / sizeof(kPoints[0]));

static double now_ms() { return (double)ggml_time_us() / 1000.0; }

static size_t gpu_used_mb() {
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return 0;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    return (total_b - free_b) / (1024 * 1024);
}

int main(int argc, char ** argv) {
    std::string model_path, video_path = "data/test_video.mp4";
    std::string inst_arg = "1,4,8,16";
    int n_frames = 4, n_threads = 4;
    bool use_gpu = true;
    sam3_debug_options debug;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"     && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--video"     && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--instances" && i + 1 < argc) inst_arg   = argv[++i];
        else if (arg == "--n-frames"  && i + 1 < argc) n_frames   = atoi(argv[++i]);
        else if (arg == "--threads"   && i + 1 < argc) n_threads  = atoi(argv[++i]);
        else if (arg == "--cpu") use_gpu = false;
        else if (sam3_example_debug_flag(argv[i], debug)) { /* diagnostics flag consumed */ }
        else if (arg == "--help" || arg == "-h") {
            printf("usage: %s --model <path.gguf> [--instances 1,4,8,16] "
                   "[--n-frames 4] [--video <path>] [--threads 4] [--cpu]\n"
                   "  diagnostics: --census [--census=2] --profile-prop --pcs-prof "
                   "--encode-timing --dump-vit-blocks --sam2-dump-dir=DIR\n",
                   argv[0]);
            return 0;
        }
    }
    if (model_path.empty()) { fprintf(stderr, "error: --model is required\n"); return 1; }

    std::vector<int> counts;
    for (size_t pos = 0; pos <= inst_arg.size();) {
        size_t comma = inst_arg.find(',', pos);
        std::string tok = inst_arg.substr(pos, comma == std::string::npos
                                               ? std::string::npos : comma - pos);
        if (!tok.empty()) counts.push_back(atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    printf("%-14s %6s %10s %12s %12s %10s\n",
           "model", "inst", "seed_ms", "track_ms/fr", "dets/frame", "vram_mb");

    for (int N : counts) {
        if (N < 1) continue;
        const int n_seed = (N < kNumPoints) ? N : kNumPoints;

        sam3_params params;
        params.model_path = model_path.c_str();
        params.use_gpu    = use_gpu;
        params.n_threads  = n_threads;
        params.debug      = debug;
        auto model = sam3_load_model(params);
        if (!model) { fprintf(stderr, "load failed for %s\n", model_path.c_str()); return 1; }
        auto state = sam3_create_state(*model, params);

        sam3_video_params vp;
        vp.hotstart_delay = 0;
        vp.max_keep_alive = 100;
        auto tracker = sam3_create_tracker(*model, vp);
        if (!tracker) { fprintf(stderr, "tracker failed\n"); return 1; }

        auto frame0 = sam3_decode_video_frame(video_path, 0);
        if (frame0.data.empty()) { fprintf(stderr, "decode failed\n"); return 1; }
        if (!sam3_encode_image(*state, *model, frame0)) return 1;

        // Seed N instances (point prompts on distinct objects).
        double t0 = now_ms();
        int added = 0;
        long seed_fg_sum = 0;
        for (int i = 0; i < n_seed; ++i) {
            sam3_pvs_params pvs;
            pvs.pos_points.push_back({kPoints[i][0], kPoints[i][1]});
            pvs.multimask = false;
            const int id = sam3_tracker_add_instance(*tracker, *state, *model, pvs, 0);
            if (id < 0) continue;
            ++added;
        }
        const double seed_ms = now_ms() - t0;

        // Track frames 1..K-1.
        double track_sum = 0.0;
        int tracks = 0;
        size_t dets_last = 0;
        for (int f = 1; f < n_frames; ++f) {
            auto frame = sam3_decode_video_frame(video_path, f);
            if (frame.data.empty()) { fprintf(stderr, "decode %d failed\n", f); return 1; }
            t0 = now_ms();
            auto r = sam3_track_frame(*tracker, *state, *model, frame);
            track_sum += now_ms() - t0;
            ++tracks;
            dets_last = r.detections.size();
            long fg = 0;
            for (const auto & d : r.detections)
                for (uint8_t v : d.mask.data) fg += (v > 127);
            seed_fg_sum += fg;
        }
        const double track_avg = tracks ? track_sum / tracks : 0.0;
        const size_t vram = gpu_used_mb();
        printf("%-14s %6d %10.0f %12.0f %12zu %10zu\n",
               model_path.c_str(), N, seed_ms, track_avg, dets_last, vram);
        fprintf(stderr, "  [inst %2d] added %d, total fg px (all frames) = %ld\n",
                N, added, seed_fg_sum);

        tracker.reset();
        state.reset();
        sam3_free_model(*model);
    }
    return 0;
}
