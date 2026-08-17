// sam3_quantize — Quantize SAM3/SAM2 model weights (GGUF) from F32/F16 to Q4_0/Q4_1/Q8_0
//
// Standard GGUF in / out, using the same gguf_init_from_file / gguf_add_tensor
// / gguf_write_to_file API family as the rest of the ecosystem. All metadata
// (arch, hparams, embedded tokenizer) is copied through with gguf_set_kv, so
// the output loads identically through sam3_load_model.
//
// Usage: sam3_quantize <input.gguf> <output.gguf> <type>
//   types: q4_0, q4_1, q8_0

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool sam3_quantize_model(const std::string & fname_inp,
                                const std::string & fname_out,
                                ggml_type qtype) {
    // ── Read source GGUF (weights loaded into a CPU ctx) ─────────────────
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/&ctx };
    struct gguf_context * src = gguf_init_from_file(fname_inp.c_str(), params);
    if (!src) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(src);
    fprintf(stderr, "%s: %lld tensors, ftype %d -> %s\n", __func__,
            (long long) n_tensors,
            gguf_find_key(src, "sam3.ftype") >= 0
                ? gguf_get_val_i32(src, gguf_find_key(src, "sam3.ftype")) : 1,
            ggml_type_name(qtype));

    // ── Create output GGUF: copy all metadata, bump the weight-type KV ───
    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, src);                                   // arch/hparams/tokenizer
    gguf_set_val_i32(out, "sam3.ftype", (int32_t) qtype);    // loader's weight_type

    const int blk_size = ggml_blck_size(qtype);

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    int    n_quantized    = 0;
    int    n_total        = 0;

    std::vector<float>       data_f32;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<uint8_t>     work;

    // gguf_set_tensor_data() stores a pointer, not a copy, and the data is
    // only dereferenced at gguf_write_to_file() time — so quantized output
    // must live in stable memory that never reallocs. Collect every quantized
    // blob into one pre-reserved pool (reserve the full source data size,
    // which is an upper bound) and point the GGUF tensors into it.
    size_t pool_size = 0;
    for (int64_t t = 0; t < n_tensors; ++t) {
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, gguf_get_tensor_name(src, t));
        if (tensor) pool_size += ggml_nbytes(tensor);
    }
    std::vector<uint8_t> out_pool;
    out_pool.reserve(pool_size);

    for (int64_t t = 0; t < n_tensors; ++t) {
        const char * name = gguf_get_tensor_name(src, t);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        if (!tensor || !tensor->data) {
            fprintf(stderr, "%s: tensor '%s' has no data\n", __func__, name);
            gguf_free(out);
            gguf_free(src);
            ggml_free(ctx);
            return false;
        }

        const int64_t n_el   = ggml_nelements(tensor);
        const ggml_type file_type = tensor->type;

        // Decide whether to quantize. This must mirror the register_* macros
        // in sam3.cpp: T2/T3/T4 store WTYPE when ne[0] is block-aligned, while
        // T1f (biases, norm params) and the T*f embedding variants always use
        // F32. GGUF n_dims cannot distinguish a [256,1] Linear output weight
        // (quantizable, registered via T2) from a [256] bias (never quantized),
        // so use the name to separate them — same decision the legacy
        // converter/quantizer made from the PyTorch ndim.
        auto name_contains = [&](const char * sub) {
            return strstr(name, sub) != nullptr;
        };
        // Match tensors registered as F32 in sam3_register_tensors (T2f/T3f/T4f).
        // These are embeddings, lookup tables, positional encodings, and special tokens
        // that must NOT be quantized.  Be specific to avoid catching weight matrices
        // like bbox_embed, mask_embed, or boxRPB_embed which ARE quantizable.
        const bool is_embedding =
            name_contains("token_embed")   || name_contains("pos_embed")
         || name_contains("query_embed")   || name_contains("label_embed")
         || name_contains("cls_embed")     || name_contains("point_embeddings")
         || name_contains("not_a_point_embed") || name_contains("no_mask_embed")
         || name_contains("no_mem_embed")  || name_contains("no_obj_embed")
         || name_contains("presence_token.weight")
         || name_contains("iou_token")     || name_contains("mask_tokens")
         || name_contains("obj_score_token")
         || name_contains("pe_gaussian")   || name_contains("freqs_cis")
         || name_contains("gamma")         || name_contains("tpos_enc")
         || name_contains("no_obj_ptr")    || name_contains("no_mem_pos_enc")
         || name_contains("trk_mask_ds")   || name_contains("latents");
        // 1D parameters (biases, layer-norm scale/shift) are registered as F32
        // via T1f and must stay unquantized.
        const bool is_1d_param = name_contains(".bias") || name_contains("norm");
        const bool quantize = (tensor->ne[0] % blk_size == 0) &&
                              !ggml_is_quantized(file_type) &&
                              !is_embedding &&
                              !is_1d_param;

        gguf_add_tensor(out, tensor);   // registers name/shape/type + offset

        if (quantize) {
            // Convert to F32, then quantize with ggml_quantize_chunk
            if (file_type == GGML_TYPE_F16) {
                data_f16.resize(n_el);
                memcpy(data_f16.data(), tensor->data, n_el * sizeof(ggml_fp16_t));
                data_f32.resize(n_el);
                for (int64_t i = 0; i < n_el; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(n_el);
                memcpy(data_f32.data(), tensor->data, n_el * sizeof(float));
            }

            const int64_t n_rows     = n_el / tensor->ne[0];
            const size_t  out_row_sz = ggml_row_size(qtype, tensor->ne[0]);
            work.resize(n_rows * out_row_sz);

            const size_t cur_size = ggml_quantize_chunk(
                qtype, data_f32.data(), work.data(), 0, n_rows, tensor->ne[0], nullptr);

            gguf_set_tensor_type(out, name, qtype);
            const size_t pool_off = out_pool.size();
            out_pool.insert(out_pool.end(), work.data(), work.data() + cur_size);
            gguf_set_tensor_data(out, name, out_pool.data() + pool_off);

            total_size_new += cur_size;
            n_quantized++;

            printf("%64s - [%5lld, %5lld, %5lld, %5lld] %6s -> %6s  %8.2f MB -> %8.2f MB\n",
                   name, (long long) tensor->ne[0], (long long) tensor->ne[1],
                   (long long) tensor->ne[2], (long long) tensor->ne[3],
                   ggml_type_name(file_type), ggml_type_name(qtype),
                   ggml_nbytes(tensor) / (1024.0 * 1024.0),
                   cur_size  / (1024.0 * 1024.0));
        } else {
            // Copy tensor as-is (data pointer stays valid until gguf_write)
            gguf_set_tensor_data(out, name, tensor->data);

            total_size_new += ggml_nbytes(tensor);

            printf("%64s - [%5lld, %5lld, %5lld, %5lld] %6s  (kept)  %8.2f MB\n",
                   name, (long long) tensor->ne[0], (long long) tensor->ne[1],
                   (long long) tensor->ne[2], (long long) tensor->ne[3],
                   ggml_type_name(file_type),
                   ggml_nbytes(tensor) / (1024.0 * 1024.0));
        }

        total_size_org += n_el * sizeof(float);
        n_total++;
    }

    const bool ok = gguf_write_to_file(out, fname_out.c_str(), false);
    if (!ok) {
        fprintf(stderr, "%s: failed to write '%s'\n", __func__, fname_out.c_str());
    }

    printf("\n");
    printf("%s: quantized %d / %d tensors\n", __func__, n_quantized, n_total);
    printf("%s: original size  = %8.2f MB (F32 equivalent)\n",
           __func__, total_size_org / (1024.0 * 1024.0));
    printf("%s: quantized size = %8.2f MB\n",
           __func__, total_size_new / (1024.0 * 1024.0));
    printf("%s: compression    = %.2fx\n",
           __func__, (double)total_size_org / total_size_new);

    gguf_free(out);
    gguf_free(src);
    ggml_free(ctx);

    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        fprintf(stderr, "  types: q4_0, q4_1, q8_0\n");
        return 1;
    }

    const std::string fname_inp  = argv[1];
    const std::string fname_out  = argv[2];
    const std::string type_str   = argv[3];

    ggml_type qtype;
    if      (type_str == "q4_0") qtype = GGML_TYPE_Q4_0;
    else if (type_str == "q4_1") qtype = GGML_TYPE_Q4_1;
    else if (type_str == "q8_0") qtype = GGML_TYPE_Q8_0;
    else {
        fprintf(stderr, "unknown type: %s\n", type_str.c_str());
        return 1;
    }

    return sam3_quantize_model(fname_inp, fname_out, qtype) ? 0 : 1;
}
