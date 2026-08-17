/**
 * profile_sam3 — per-stage latency profiler for the SAM3 ViT image encoder.
 *
 * Loads a SAM3/SAM2 model and times each encoder sub-stage (patch embed,
 * layer norm, window part/unpart, QKV/proj GEMMs, attention core, MLP) as a
 * separate sub-graph on the selected backend, then prints a breakdown table
 * and an estimated full-encode latency.
 *
 * Usage:
 *   profile_sam3 [options]
 *
 * Options:
 *   --model <path>      Model file      (default: models/sam3-visual-f16.gguf)
 *   --device <dev>      auto|cpu|cuda|vulkan (default: auto)
 *   --n-threads <n>     CPU threads      (default: 4)
 *   --n-warmup <n>      Warmup iterations (default: 2)
 *   --n-iter <n>        Timed iterations  (default: 5)
 *   --block <idx>       Only profile this block (default: all)
 */

#include "sam3.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static const char * stage_name(sam3_vit_block_stage s) {
    switch (s) {
        case SAM3_VIT_BLOCK_STAGE_NORM1:         return "norm1";
        case SAM3_VIT_BLOCK_STAGE_WINDOW_PART:   return "win_part";
        case SAM3_VIT_BLOCK_STAGE_QKV_PROJ:      return "qkv_proj";
        case SAM3_VIT_BLOCK_STAGE_ATTN_CORE:     return "attn_core";
        case SAM3_VIT_BLOCK_STAGE_ATTN_PROJ:     return "attn_proj";
        case SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART: return "win_unpart";
        case SAM3_VIT_BLOCK_STAGE_NORM2:         return "norm2";
        case SAM3_VIT_BLOCK_STAGE_MLP_FC1:       return "mlp_fc1";
        case SAM3_VIT_BLOCK_STAGE_MLP_GELU:      return "mlp_gelu";
        case SAM3_VIT_BLOCK_STAGE_MLP_FC2:       return "mlp_fc2";
        case SAM3_VIT_BLOCK_STAGE_MLP:           return "mlp_full";
    }
    return "?";
}

static const char * prefix_name(sam3_vit_prefix_stage s) {
    switch (s) {
        case SAM3_VIT_PREFIX_STAGE_PATCH_EMBED:     return "patch_embed";
        case SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL:    return "patch_im2col";
        case SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW:return "patch_mulmat_raw";
        case SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT:    return "patch_mulmat";
        case SAM3_VIT_PREFIX_STAGE_POS_ADD:         return "pos_add";
        case SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM:     return "ln_pre_norm";
        case SAM3_VIT_PREFIX_STAGE_LN_PRE:          return "ln_pre";
    }
    return "?";
}

int main(int argc, char ** argv) {
    std::string model_path = "models/sam3-visual-f16.gguf";
    sam3_device device     = SAM3_DEVICE_AUTO;
    int n_threads = 4, n_warmup = 2, n_iter = 5, only_block = -1;

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--model")   == 0 && i + 1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "--device")  == 0 && i + 1 < argc) {
            std::string d = argv[++i];
            if      (d == "auto")   device = SAM3_DEVICE_AUTO;
            else if (d == "cpu")    device = SAM3_DEVICE_CPU;
            else if (d == "cuda")   device = SAM3_DEVICE_CUDA;
            else if (d == "vulkan") device = SAM3_DEVICE_VULKAN;
            else { fprintf(stderr, "unknown --device '%s'\n", d.c_str()); return 1; }
        }
        else if (strcmp(argv[i], "--n-threads") == 0 && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--n-warmup")  == 0 && i + 1 < argc) { n_warmup  = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--n-iter")    == 0 && i + 1 < argc) { n_iter    = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--block")     == 0 && i + 1 < argc) { only_block = atoi(argv[++i]); }
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu    = true;
    params.n_threads  = n_threads;
    params.device     = device;

    fprintf(stderr, "Loading %s ...\n", model_path.c_str());
    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    fprintf(stderr, "backend: %s\n\n", sam3_backend_name(*model));

    // Infer network geometry from a few known weight tensors
    sam3_tensor_info ti;
    int patch = 14, E = 0, D = 0, grid = 0, n_blocks = 0;
    if (sam3_get_model_tensor_info(*model, "vit.patch_embed.proj.weight", ti)) {
        patch = (int)ti.ne[0]; E = (int)ti.ne[3];
    }
    if (sam3_get_model_tensor_info(*model, "vit.pos_embed", ti)) {
        grid = (int)ti.ne[1];
    }
    if (sam3_get_model_tensor_info(*model, "vit.blocks.0.mlp.fc1.weight", ti)) {
        D = (int)ti.ne[0];
    }
    if (sam3_get_model_tensor_info(*model, "vit.blocks.0.qkv.weight", ti)) {
        n_blocks = 32; // per-SAM3 layout; refined below via block probe
    }
    if (grid <= 0 || E <= 0) {
        fprintf(stderr, "error: cannot infer network geometry from model tensors\n");
        return 1;
    }
    const int img_size = grid * patch;
    // discover the block count by probing until a block weight is missing
    for (n_blocks = 1; n_blocks < 128; ++n_blocks) {
        char name[64];
        snprintf(name, sizeof(name), "vit.blocks.%d.qkv.weight", n_blocks);
        if (!sam3_get_model_tensor_info(*model, name, ti)) break;
    }
    fprintf(stderr, "img=%d patch=%d E=%d mlp=%d grid=%d n_blocks=%d\n\n",
            img_size, patch, E, D, grid, n_blocks);

    // ── Prefix stages ────────────────────────────────────────────────────
    int64_t img_ne[4] = { img_size, img_size, 3, 1 };
    std::vector<float> img(img_size * img_size * 3, 0.001f);

    struct PrefixStat { std::string name; double ms; int64_t ne[4]; };
    std::vector<PrefixStat> prefix_stats;

    for (int s = (int)SAM3_VIT_PREFIX_STAGE_PATCH_EMBED; s <= (int)SAM3_VIT_PREFIX_STAGE_LN_PRE; ++s) {
        std::vector<float> out;
        int64_t out_ne[4] = {0,0,0,0};
        const float * in = img.data();
        int64_t in_ne[4]; std::copy(std::begin(img_ne), std::end(img_ne), in_ne);

        if (!sam3_test_run_vit_prefix_stage(*model, (sam3_vit_prefix_stage)s, in, in_ne, out, out_ne, n_threads)) {
            fprintf(stderr, "  %-16s : SKIP (unsupported)\n", prefix_name((sam3_vit_prefix_stage)s));
            continue;
        }
        for (int it = 0; it < n_warmup; ++it) {
            sam3_test_run_vit_prefix_stage(*model, (sam3_vit_prefix_stage)s, in, in_ne, out, out_ne, n_threads);
        }
        double t0 = now_ms();
        for (int it = 0; it < n_iter; ++it) {
            sam3_test_run_vit_prefix_stage(*model, (sam3_vit_prefix_stage)s, in, in_ne, out, out_ne, n_threads);
        }
        double dt = (now_ms() - t0) / n_iter;
        fprintf(stderr, "  %-16s : %8.2f ms   out [%lld,%lld,%lld,%lld]\n",
                prefix_name((sam3_vit_prefix_stage)s), dt,
                (long long)out_ne[0], (long long)out_ne[1], (long long)out_ne[2], (long long)out_ne[3]);
        prefix_stats.push_back({prefix_name((sam3_vit_prefix_stage)s), dt, {out_ne[0],out_ne[1],out_ne[2],out_ne[3]}});

        // chain the output as the next stage's input where shapes allow
        std::copy(std::begin(out_ne), std::end(out_ne), std::begin(img_ne));
        img.swap(out);
        img.resize((size_t)img_ne[0]*img_ne[1]*img_ne[2]*img_ne[3], 0.0f);
    }

    // ── Block stages ─────────────────────────────────────────────────────
    struct StageStat { std::string name; double ms_sum; int n; };
    std::vector<StageStat> stage_stats;
    auto ss = [&](const char * n) -> StageStat & {
        for (auto & st : stage_stats) if (st.name == n) return st;
        stage_stats.push_back({n, 0.0, 0});
        return stage_stats.back();
    };

    const int b_lo = only_block >= 0 ? only_block : 0;
    const int b_hi = only_block >= 0 ? only_block : n_blocks - 1;

    // feature grid for the first block: [E, W, W] (windowed part for later blocks)
    std::vector<float> feat((size_t)E * grid * grid, 0.001f);
    int64_t feat_ne[4] = { E, grid, grid, 1 };

    for (int b = b_lo; b <= b_hi; ++b) {
        std::vector<float> x = feat;
        int64_t ne[4]; std::copy(std::begin(feat_ne), std::end(feat_ne), ne);

        for (int s = 0; s <= (int)SAM3_VIT_BLOCK_STAGE_MLP; ++s) {
            if (s == (int)SAM3_VIT_BLOCK_STAGE_MLP && only_block < 0) break; // MLP full only on demand
            std::vector<float> out;
            int64_t out_ne[4] = {0,0,0,0};
            if (!sam3_test_run_vit_block_stage(*model, b, (sam3_vit_block_stage)s,
                                               x.data(), ne, out, out_ne, n_threads)) {
                fprintf(stderr, "  block %2d %-10s : SKIP\n", b, stage_name((sam3_vit_block_stage)s));
                continue;
            }
            for (int it = 0; it < n_warmup; ++it) {
                sam3_test_run_vit_block_stage(*model, b, (sam3_vit_block_stage)s,
                                              x.data(), ne, out, out_ne, n_threads);
            }
            double t0 = now_ms();
            for (int it = 0; it < n_iter; ++it) {
                sam3_test_run_vit_block_stage(*model, b, (sam3_vit_block_stage)s,
                                              x.data(), ne, out, out_ne, n_threads);
            }
            double dt = (now_ms() - t0) / n_iter;
            fprintf(stderr, "  block %2d %-10s : %8.2f ms\n", b, stage_name((sam3_vit_block_stage)s), dt);
            ss(stage_name((sam3_vit_block_stage)s)).ms_sum += dt;
            ss(stage_name((sam3_vit_block_stage)s)).n += 1;

            // chain output as next stage's input (window part/unpart change shape)
            std::copy(std::begin(out_ne), std::end(out_ne), std::begin(ne));
            x.swap(out);
            x.resize((size_t)ne[0]*ne[1]*ne[2]*ne[3], 0.0f);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────
    fprintf(stderr, "\n=== SUMMARY (avg over %d blocks) ===\n", b_hi - b_lo + 1);
    double total = 0;
    for (const auto & ps : prefix_stats) { fprintf(stderr, "  %-16s : %8.2f ms\n", ps.name.c_str(), ps.ms); total += ps.ms; }
    for (const auto & st : stage_stats) {
        double avg = st.ms_sum / st.n;
        fprintf(stderr, "  %-16s : %8.2f ms\n", st.name.c_str(), avg);
        total += avg * n_blocks;
    }
    fprintf(stderr, "  ----------------------------\n");
    fprintf(stderr, "  %-16s : %8.2f ms  (estimated full encoder)\n", "TOTAL", total);
    return 0;
}
