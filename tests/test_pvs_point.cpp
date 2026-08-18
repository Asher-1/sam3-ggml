/**
 * Quick PVS test: single positive point on the llama image.
 * Dumps mask to file for visual inspection.
 */
#include "sam3.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double elapsed_ms(const std::chrono::high_resolution_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
}

static double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) * 0.5
        : values[middle];
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <model.gguf> <image.jpg> [x y] "
                "[--device auto|cpu|cuda|vulkan] [--warmup N] [--repeat N]\n",
                argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    float px = 315.0f;
    float py = 250.0f;
    sam3_device device = SAM3_DEVICE_CPU;
    int warmup = 0;
    int repeat = 1;

    int arg_index = 3;
    if (arg_index < argc && argv[arg_index][0] != '-') {
        if (arg_index + 1 >= argc || argv[arg_index + 1][0] == '-') {
            fprintf(stderr, "Both x and y coordinates are required\n");
            return 1;
        }
        px = atof(argv[arg_index++]);
        py = atof(argv[arg_index++]);
    }

    for (int i = arg_index; i < argc; ++i) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "auto") device = SAM3_DEVICE_AUTO;
            else if (value == "cpu") device = SAM3_DEVICE_CPU;
            else if (value == "cuda") device = SAM3_DEVICE_CUDA;
            else if (value == "vulkan") device = SAM3_DEVICE_VULKAN;
            else {
                fprintf(stderr, "Unknown device: %s\n", value.c_str());
                return 1;
            }
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (warmup < 0 || repeat < 1) {
        fprintf(stderr, "--warmup must be >= 0 and --repeat must be >= 1\n");
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.device = device;
    params.use_gpu = device != SAM3_DEVICE_CPU;
    params.n_threads = 4;

    fprintf(stderr, "Loading model...\n");
    auto load_start = std::chrono::high_resolution_clock::now();
    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    fprintf(stderr, "Backend: %s, load: %.3f ms\n",
            sam3_backend_name(*model), elapsed_ms(load_start));

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    fprintf(stderr, "Loading image: %s\n", image_path.c_str());
    auto image = sam3_load_image(image_path);
    if (image.data.empty()) { fprintf(stderr, "Failed to load image\n"); return 1; }
    fprintf(stderr, "Image: %dx%d\n", image.width, image.height);

    std::vector<double> encode_times;
    encode_times.reserve(repeat);
    for (int i = 0; i < warmup + repeat; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if (!sam3_encode_image_pvs(*state, *model, image)) {
            fprintf(stderr, "Failed to encode image\n"); return 1;
        }
        const double ms = elapsed_ms(start);
        fprintf(stderr, "Encode %s %d: %.3f ms\n",
                i < warmup ? "warmup" : "timed", i < warmup ? i + 1 : i - warmup + 1, ms);
        if (i >= warmup) encode_times.push_back(ms);
    }

    fprintf(stderr, "\n═══ PVS: point at (%.1f, %.1f) ═══\n", px, py);

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = false;

    std::vector<double> segment_times;
    segment_times.reserve(repeat);
    sam3_result result;
    for (int i = 0; i < warmup + repeat; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        result = sam3_segment_pvs(*state, *model, pvs);
        const double ms = elapsed_ms(start);
        fprintf(stderr, "Segment %s %d: %.3f ms\n",
                i < warmup ? "warmup" : "timed", i < warmup ? i + 1 : i - warmup + 1, ms);
        if (i >= warmup) segment_times.push_back(ms);
    }
    double encode_sum = 0.0;
    for (double ms : encode_times) encode_sum += ms;
    double segment_sum = 0.0;
    for (double ms : segment_times) segment_sum += ms;
    const double encode_avg = encode_sum / encode_times.size();
    const double segment_avg = segment_sum / segment_times.size();
    fprintf(stderr, "Timing: encode_avg=%.3f ms segment_avg=%.3f ms inference=%.3f ms\n",
            encode_avg, segment_avg, encode_avg + segment_avg);
    const double encode_median = median_ms(encode_times);
    const double segment_median = median_ms(segment_times);
    fprintf(stderr, "Timing p50: encode=%.3f ms segment=%.3f ms inference=%.3f ms\n",
            encode_median, segment_median, encode_median + segment_median);
    fprintf(stderr, "Result: %zu detections\n", result.detections.size());

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& d = result.detections[i];
        fprintf(stderr, "  det %zu: score=%.4f iou=%.4f obj=%.4f box=[%.1f,%.1f,%.1f,%.1f] mask=%dx%d\n",
                i, d.score, d.iou_score, d.mask.obj_score,
                d.box.x0, d.box.y0, d.box.x1, d.box.y1,
                d.mask.width, d.mask.height);

        std::string mask_path = "/tmp/pvs_mask_" + std::to_string(i) + ".png";
        sam3_save_mask(d.mask, mask_path);
        fprintf(stderr, "  Saved: %s\n", mask_path.c_str());

        // Count mask pixels
        int n_on = 0;
        for (size_t j = 0; j < d.mask.data.size(); ++j)
            if (d.mask.data[j] > 0) n_on++;
        fprintf(stderr, "  Mask: %d/%zu pixels on (%.1f%%)\n",
                n_on, d.mask.data.size(), 100.0f * n_on / d.mask.data.size());
    }

    sam3_free_model(*model);
    return 0;
}
