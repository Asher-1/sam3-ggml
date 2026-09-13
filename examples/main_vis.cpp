// sam3_vis — Offline segmentation visualization (no SDL2/ImGui required)
//
// C++ counterpart of the official example notebooks' visualization flow:
// run segmentation, then render a full-scene PNG with per-instance colored
// mask overlay, box outlines and "id score" labels (sam3_render_result).
//
// Usage (image):
//   sam3_vis --model <path.gguf> --image <path> [--prompt "text"]
//            [--point x,y] [--neg-point x,y] [--box x0,y0,x1,y1]
//            [--exemplar x0,y0,x1,y1] [--neg-exemplar x0,y0,x1,y1]
//            [--iom 0.3] [--out <png>] [--threads N]
//            [--device auto|cpu|cuda|vulkan] [--alpha 0.35]
//            [--no-boxes] [--no-labels]
//
//   Prompt selection mirrors the official API surface:
//     --prompt            → text-prompted detection (PCS)
//     --point/--box/...   → point/box segmentation (PVS)
//
// Usage (video): render every tracked frame to <out-dir>/frame_%05d.png
//   sam3_vis --model <path.gguf> --video <path> --prompt "text"
//            [--out-dir <dir>] [--max-frames N] [common options]
//   Without --prompt (visual-only tracking) a first-frame --point or --box
//   seeds one instance, then frames are propagated.

#include "sam3.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

bool parse_float2(const char* s, float& a, float& b) {
    return sscanf(s, "%f,%f", &a, &b) == 2;
}

bool parse_float4(const char* s, float& a, float& b, float& c, float& d) {
    return sscanf(s, "%f,%f,%f,%f", &a, &b, &c, &d) == 4;
}

// Best-effort directory creation (frame output dir); errors ignored — the
// subsequent PNG write reports the real failure if creation did not stick.
void ensure_dir(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

void usage() {
    fprintf(stderr,
        "sam3_vis — offline segmentation visualization\n\n"
        "  --model <path>          GGUF model (required)\n"
        "  --image <path>          input image (image mode)\n"
        "  --video <path>          input video (video mode)\n"
        "  --prompt <text>         text prompt -> PCS detection (SAM 3)\n"
        "  --point x,y             positive point (PVS), repeatable\n"
        "  --neg-point x,y         negative point (PVS), repeatable\n"
        "  --box x0,y0,x1,y1       box prompt (PVS)\n"
        "  --exemplar x0,y0,x1,y1  positive exemplar box (PCS), repeatable\n"
        "  --neg-exemplar ...      negative exemplar box (PCS), repeatable\n"
        "  --iom <f>               IoM mask dedup threshold (0 = off, official agent default 0.3)\n"
        "  --out <png>             output image path (default vis.png)\n"
        "  --out-dir <dir>         output directory for video frames (default vis_frames)\n"
        "  --max-frames <n>        limit video frames (0 = all)\n"
        "  --threads <n>           CPU threads (default 4)\n"
        "  --device <d>            auto|cpu|cuda|vulkan (default auto)\n"
        "  --alpha <f>             mask overlay opacity (default 0.35)\n"
        "  --no-boxes / --no-labels\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, image_path, video_path, text_prompt, out_path = "vis.png";
    std::string out_dir = "vis_frames";
    std::vector<sam3_point> pos_points, neg_points;
    std::vector<sam3_box> pos_exemplars, neg_exemplars;
    sam3_box box = {0, 0, 0, 0};
    bool use_box = false;
    float iom = 0.0f, alpha = 0.35f;
    int n_threads = 4, max_frames = 0;
    sam3_device device = SAM3_DEVICE_AUTO;
    bool draw_boxes = true, draw_labels = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](void) -> const char* {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if      (arg == "--model")   { auto v = next(); if (v) model_path = v; }
        else if (arg == "--image")   { auto v = next(); if (v) image_path = v; }
        else if (arg == "--video")   { auto v = next(); if (v) video_path = v; }
        else if (arg == "--prompt")  { auto v = next(); if (v) text_prompt = v; }
        else if (arg == "--point") {
            auto v = next(); sam3_point p;
            if (v && parse_float2(v, p.x, p.y)) pos_points.push_back(p);
        } else if (arg == "--neg-point") {
            auto v = next(); sam3_point p;
            if (v && parse_float2(v, p.x, p.y)) neg_points.push_back(p);
        } else if (arg == "--box") {
            auto v = next();
            if (v && parse_float4(v, box.x0, box.y0, box.x1, box.y1)) use_box = true;
        } else if (arg == "--exemplar") {
            auto v = next(); sam3_box b;
            if (v && parse_float4(v, b.x0, b.y0, b.x1, b.y1)) pos_exemplars.push_back(b);
        } else if (arg == "--neg-exemplar") {
            auto v = next(); sam3_box b;
            if (v && parse_float4(v, b.x0, b.y0, b.x1, b.y1)) neg_exemplars.push_back(b);
        } else if (arg == "--iom")       { auto v = next(); if (v) iom = (float)atof(v); }
        else if (arg == "--out")         { auto v = next(); if (v) out_path = v; }
        else if (arg == "--out-dir")     { auto v = next(); if (v) out_dir = v; }
        else if (arg == "--max-frames")  { auto v = next(); if (v) max_frames = atoi(v); }
        else if (arg == "--threads")     { auto v = next(); if (v) n_threads = atoi(v); }
        else if (arg == "--device") {
            auto v = next();
            if (!v) continue;
            if      (!strcmp(v, "cpu"))    device = SAM3_DEVICE_CPU;
            else if (!strcmp(v, "cuda"))   device = SAM3_DEVICE_CUDA;
            else if (!strcmp(v, "vulkan")) device = SAM3_DEVICE_VULKAN;
        }
        else if (arg == "--alpha")   { auto v = next(); if (v) alpha = (float)atof(v); }
        else if (arg == "--no-boxes")  draw_boxes = false;
        else if (arg == "--no-labels") draw_labels = false;
        else { fprintf(stderr, "unknown argument: %s\n", arg.c_str()); usage(); return 1; }
    }

    if (model_path.empty() || (image_path.empty() == video_path.empty())) {
        fprintf(stderr, "error: --model plus exactly one of --image/--video required\n");
        usage();
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.n_threads  = n_threads;
    params.device     = device;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "model: %s (backend: %s)\n", model_path.c_str(),
            sam3_backend_name(*model));

    sam3_vis_params vp;
    vp.mask_alpha  = alpha;
    vp.draw_boxes  = draw_boxes;
    vp.draw_labels = draw_labels;

    const bool use_pcs = !text_prompt.empty() || !pos_exemplars.empty() ||
                         !neg_exemplars.empty();

    if (!image_path.empty()) {
        sam3_image image = sam3_load_image(image_path);
        if (image.data.empty()) {
            fprintf(stderr, "error: failed to load image '%s'\n", image_path.c_str());
            return 1;
        }
        auto state = sam3_create_state(*model, params);

        sam3_result result;
        if (use_pcs) {
            if (sam3_is_visual_only(*model)) {
                fprintf(stderr, "error: PCS needs a full SAM 3 checkpoint "
                                "(exemplar-only prompts are not supported on visual-only models)\n");
                return 1;
            }
            // Full SAM3 path: the detector consumes the detector neck too.
            if (!sam3_encode_image(*state, *model, image)) {
                fprintf(stderr, "error: image encoding failed\n");
                return 1;
            }
            sam3_pcs_params pcs;
            pcs.text_prompt    = text_prompt;
            pcs.pos_exemplars  = pos_exemplars;
            pcs.neg_exemplars  = neg_exemplars;
            result = sam3_segment_pcs(*state, *model, pcs);
        } else {
            // PVS path: skip the detector neck for full SAM3 checkpoints.
            if (!sam3_encode_image_pvs(*state, *model, image)) {
                fprintf(stderr, "error: image encoding failed\n");
                return 1;
            }
            sam3_pvs_params pvs;
            pvs.pos_points = pos_points;
            pvs.neg_points = neg_points;
            pvs.box     = box;
            pvs.use_box = use_box;
            result = sam3_segment_pvs(*state, *model, pvs);
        }

        if (iom > 0.0f) {
            const int removed = sam3_remove_overlapping_masks(result, iom);
            fprintf(stderr, "IoM dedup (thr %.2f): removed %d / %zu detections\n",
                    iom, removed, removed + result.detections.size());
        }

        fprintf(stderr, "detections: %zu\n", result.detections.size());
        for (const auto& det : result.detections) {
            fprintf(stderr, "  id %d score %.3f box [%.0f,%.0f,%.0f,%.0f]\n",
                    det.instance_id, det.score,
                    det.box.x0, det.box.y0, det.box.x1, det.box.y1);
        }

        sam3_image vis = sam3_render_result(image, result, vp);
        if (!sam3_save_image(vis, out_path)) {
            fprintf(stderr, "error: failed to write '%s'\n", out_path.c_str());
            return 1;
        }
        fprintf(stderr, "saved: %s\n", out_path.c_str());
        return 0;
    }

    // ── Video mode ─────────────────────────────────────────────────────────
    ensure_dir(out_dir);
    sam3_video_info info = sam3_get_video_info(video_path);
    if (info.n_frames <= 0) {
        fprintf(stderr, "error: cannot probe video '%s'\n", video_path.c_str());
        return 1;
    }
    const int n_frames = (max_frames > 0) ? std::min(max_frames, info.n_frames)
                                          : info.n_frames;
    fprintf(stderr, "video: %s (%dx%d, %d frames, rendering %d)\n",
            video_path.c_str(), info.width, info.height, info.n_frames, n_frames);

    auto state = sam3_create_state(*model, params);

    sam3_tracker_ptr tracker;   // custom deleter: complete type lives in the library
    if (!text_prompt.empty()) {
        if (sam3_is_visual_only(*model)) {
            fprintf(stderr, "error: text-prompted tracking needs a full SAM 3 checkpoint\n");
            return 1;
        }
        sam3_video_params vp2;
        vp2.text_prompt = text_prompt;
        tracker = sam3_create_tracker(*model, vp2);
    } else {
        if (!use_box && pos_points.empty()) {
            fprintf(stderr, "error: visual-only tracking needs --prompt, "
                            "or a first-frame --point/--box to seed an instance\n");
            return 1;
        }
        tracker = sam3_create_visual_tracker(*model, {});
    }

    for (int f = 0; f < n_frames; ++f) {
        sam3_image frame = sam3_decode_video_frame(video_path, f);
        if (frame.data.empty()) {
            fprintf(stderr, "error: failed to decode frame %d\n", f);
            return 1;
        }

        // Frame 0 with visual-only prompts seeds one instance before propagating.
        sam3_result result;
        if (text_prompt.empty()) {
            if (f == 0) {
                if (!sam3_encode_image(*state, *model, frame)) {
                    fprintf(stderr, "error: frame 0 encoding failed\n");
                    return 1;
                }
                sam3_pvs_params pvs;
                pvs.pos_points = pos_points;
                pvs.neg_points = neg_points;
                pvs.box     = box;
                pvs.use_box = use_box;
                const int id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
                if (id < 0) {
                    fprintf(stderr, "error: failed to add tracking instance\n");
                    return 1;
                }
                result = sam3_propagate_frame(*tracker, *state, *model, frame);
            } else {
                result = sam3_propagate_frame(*tracker, *state, *model, frame);
            }
        } else {
            result = sam3_track_frame(*tracker, *state, *model, frame);
        }

        if (iom > 0.0f) sam3_remove_overlapping_masks(result, iom);

        char frame_path[512];
        snprintf(frame_path, sizeof(frame_path), "%s/frame_%05d.png",
                 out_dir.c_str(), f);
        sam3_image vis = sam3_render_result(frame, result, vp);
        if (!sam3_save_image(vis, frame_path)) {
            fprintf(stderr, "error: failed to write '%s'\n", frame_path);
            return 1;
        }
        fprintf(stderr, "frame %d/%d: %zu instances -> %s\n",
                f + 1, n_frames, result.detections.size(), frame_path);
    }

    fprintf(stderr, "done: %d frames rendered to %s/\n", n_frames, out_dir.c_str());
    return 0;
}
