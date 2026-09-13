// upgrade_probe — numerical fingerprint for ggml upgrade A/B verification.
// Usage: upgrade_probe <model.gguf> <image.jpg> <cpu|cuda> <px> <py> [repeat] [mode]
//
// Loads the model on the requested device and prints a deterministic
// fingerprint for the requested inference path:
//   mode = pvs    (default) point-prompt segmentation on the image
//   mode = pcs:<text>        text-prompted detection (full SAM 3 only)
//   mode = video:<n_frames>  seed one instance from the point prompt on the
//                            image, then propagate the same image n times
//                            (deterministic tracking chain, no video file)
// Output: backend name, detection count, per-detection score/iou/box (6
// decimals), mask size, 0/255 pixel count, and a 64-bit FNV-1a hash of the
// mask bytes. Compile it against two builds (baseline / candidate) with
// identical inputs and diff the stdout to prove the upgrade kept inference
// bit-identical (or bounded-different on CPU float reordering).

#include "sam3.h"
#include "stb_image.h"

#include <cstdint>
#include <cstddef>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> & v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void print_dets(const char * prefix, size_t idx, const sam3_detection & d) {
    uint64_t hash = fnv1a(d.mask.data.data(), d.mask.data.size());
    size_t ones = 0;
    for (uint8_t v : d.mask.data) ones += (v != 0);
    printf("  %s[%zu] score=%.6f iou=%.6f box=[%.3f %.3f %.3f %.3f] mask=%dx%d ones=%zu fnv=%016llx token0=%.6f\n",
           prefix, idx, d.score, d.iou_score, d.box.x0, d.box.y0, d.box.x1, d.box.y1,
           d.mask.width, d.mask.height, ones, (unsigned long long) hash,
           d.sam_token.empty() ? 0.0f : d.sam_token[0]);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <model.gguf> <image> <cpu|cuda> <px> <py> [repeat=5] [mode]\n"
                        "  mode: pvs (default) | pcs:<text> | video:<n_frames>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const bool  use_cuda = strcmp(argv[3], "cuda") == 0;
    const float px = atof(argv[4]);
    const float py = atof(argv[5]);
    const int   repeat = argc > 6 ? atoi(argv[6]) : 5;
    const char * mode = argc > 7 ? argv[7] : (char *)"pvs";

    // mode parsing
    enum { M_PVS, M_PCS, M_VIDEO } m = M_PVS;
    std::string text_prompt;
    int n_video_frames = 3;
    if (strncmp(mode, "pcs:", 4) == 0) {
        m = M_PCS;
        text_prompt = mode + 4;
    } else if (strncmp(mode, "video:", 6) == 0) {
        m = M_VIDEO;
        n_video_frames = atoi(mode + 6);
        if (n_video_frames < 1) n_video_frames = 1;
    }

    int w, h, comp;
    uint8_t * pixels = stbi_load(image_path.c_str(), &w, &h, &comp, 3);
    if (!pixels) {
        fprintf(stderr, "ERROR: cannot load image %s\n", image_path.c_str());
        return 1;
    }
    sam3_image img;
    img.width  = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(pixels, pixels + (size_t)w * h * 3);
    stbi_image_free(pixels);

    sam3_params params;
    params.model_path = model_path;
    params.n_threads  = 4;
    params.use_gpu    = use_cuda;
    params.device     = use_cuda ? SAM3_DEVICE_CUDA : SAM3_DEVICE_CPU;

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "ERROR: load model failed\n"); return 1; }
    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "ERROR: create state failed\n"); return 1; }

    const char * mode_prefix = (m == M_VIDEO) ? "frame" : "det";
    sam3_result res;

    if (m == M_VIDEO) {
        // ── Deterministic tracking chain: seed one instance from the point
        // prompt, then propagate the same image n_video_frames times. ──
        auto tracker = sam3_create_visual_tracker(*model, {});
        if (!tracker) { fprintf(stderr, "ERROR: create tracker failed\n"); return 1; }
        if (!sam3_encode_image(*state, *model, img)) {
            fprintf(stderr, "ERROR: encode image failed\n");
            return 1;
        }
        sam3_pvs_params seed;
        seed.pos_points.push_back({px, py});
        if (sam3_tracker_add_instance(*tracker, *state, *model, seed) < 0) {
            fprintf(stderr, "ERROR: add instance failed\n");
            return 1;
        }
        res = sam3_propagate_frame(*tracker, *state, *model, img);
        if (res.detections.empty()) {
            fprintf(stderr, "ERROR: frame 0 produced no detections\n");
            return 2;
        }
        for (size_t i = 0; i < res.detections.size(); i++)
            print_dets(mode_prefix, i, res.detections[i]);
        std::vector<double> t_prop;
        for (int f = 1; f < n_video_frames; ++f) {
            double t0 = now_ms();
            sam3_result fr = sam3_propagate_frame(*tracker, *state, *model, img);
            double t1 = now_ms();
            t_prop.push_back(t1 - t0);
            for (size_t i = 0; i < fr.detections.size(); i++)
                print_dets(mode_prefix, (size_t)f * 100 + i, fr.detections[i]);
            res = std::move(fr);
        }
        if (!t_prop.empty())
            printf("timing: propagate_med=%.1fms (n=%zu)\n",
                   median(t_prop), t_prop.size());
    } else {
        // ── Image paths: PCS (full SAM 3) or PVS (all models) ──
        if (!sam3_encode_image(*state, *model, img)) {
            fprintf(stderr, "ERROR: encode image failed\n");
            return 1;
        }

        sam3_result first;
        if (m == M_PCS) {
            sam3_pcs_params pcs;
            pcs.text_prompt = text_prompt;
            first = sam3_segment_pcs(*state, *model, pcs);
        } else {
            sam3_pvs_params pvs;
            pvs.pos_points.push_back({px, py});
            pvs.multimask = false;
            first = sam3_segment_pvs(*state, *model, pvs);
        }
        res = std::move(first);

        // Steady-state timing: repeat in-process, take the median of each stage.
        // This eliminates the fork/cold-start noise that pollutes single-shot
        // benchmarks (CUDA context init, kernel module load, page cache).
        std::vector<double> t_enc, t_seg;
        for (int r = 0; r < repeat; r++) {
            double t0 = now_ms();
            if (!sam3_encode_image(*state, *model, img)) {
                fprintf(stderr, "ERROR: encode failed on iter %d\n", r);
                return 1;
            }
            double t1 = now_ms();
            sam3_result res_r;
            if (m == M_PCS) {
                sam3_pcs_params pcs;
                pcs.text_prompt = text_prompt;
                res_r = sam3_segment_pcs(*state, *model, pcs);
            } else {
                sam3_pvs_params pvs;
                pvs.pos_points.push_back({px, py});
                pvs.multimask = false;
                res_r = sam3_segment_pvs(*state, *model, pvs);
            }
            double t2 = now_ms();
            t_enc.push_back(t1 - t0);
            t_seg.push_back(t2 - t1);
            if (r == repeat - 1) res = std::move(res_r);
        }
        printf("timing: encode_med=%.1fms seg_med=%.1fms (n=%d)\n",
               median(t_enc), median(t_seg), repeat);
    }

    printf("mode=%s backend=%s model=%s device=%s ndet=%zu\n",
           mode, sam3_backend_name(*model),
           model_path.c_str(), argv[3], res.detections.size());
    for (size_t i = 0; i < res.detections.size(); i++)
        print_dets(mode_prefix, i, res.detections[i]);
    return res.detections.empty() ? 2 : 0;
}