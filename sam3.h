#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/*
** ── DLL Export Decoration ────────────────────────────────────────────────
**
** SAM3_API mirrors LLAMA_API / SD_API: tag every public function (and the
** opaque-type deleters) so MSVC emits the correct __declspec when sam3 is
** built / consumed as a shared library. For static builds the macro expands
** to nothing and there is no ABI surface.
**
** Define SAM3_SHARED on both build and consumer side when sam3 is a DLL;
** additionally define SAM3_BUILD only on the side that is compiling sam3.cpp
** (the CMake target sets this via target_compile_definitions).
*/
#ifdef SAM3_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef SAM3_BUILD
#            define SAM3_API __declspec(dllexport)
#        else
#            define SAM3_API __declspec(dllimport)
#        endif
#    else
#        define SAM3_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define SAM3_API
#endif

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

#define SAM3_VERSION_MAJOR 1
#define SAM3_VERSION_MINOR 0
#define SAM3_VERSION_PATCH 0
#define SAM3_VERSION       "1.0.0"

/*
** ── Forward Declarations ─────────────────────────────────────────────────
*/

struct sam3_model;
struct sam3_state;
struct sam3_tracker;

/* Custom deleters so unique_ptr works with forward-declared opaque types.
** The struct itself is dllexport'd so the out-of-line operator() (which calls
** sam3_free_state / sam3_tracker_reset + delete) is reachable across the DLL
** boundary — the deleter needs the full type of sam3_state / sam3_tracker for
** `delete`, which only exists inside sam3.cpp. */
struct SAM3_API sam3_state_deleter   { void operator()(sam3_state * p) const; };
struct SAM3_API sam3_tracker_deleter { void operator()(sam3_tracker * p) const; };

using sam3_state_ptr   = std::unique_ptr<sam3_state,   sam3_state_deleter>;
using sam3_tracker_ptr = std::unique_ptr<sam3_tracker,  sam3_tracker_deleter>;

/*
** ── Model Type ──────────────────────────────────────────────────────────
*/

enum sam3_model_type {
    SAM3_MODEL_SAM3        = 0,  // Full SAM3 (ViT + detector + tracker)
    SAM3_MODEL_SAM3_VISUAL = 1,  // SAM3 visual-only (ViT + tracker, no text)
    SAM3_MODEL_SAM2        = 2,  // SAM2 (Hiera + tracker, no text/detector)
};

/*****************************************************************************
** Public Data Types
**
** Geometry primitives, images, masks, and detection results.
*****************************************************************************/

struct sam3_point {
    float x;
    float y;
};

struct sam3_box {
    float x0;  // top-left x
    float y0;  // top-left y
    float x1;  // bottom-right x
    float y1;  // bottom-right y
};

struct sam3_image {
    int width    = 0;
    int height   = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

struct sam3_mask {
    int   width       = 0;
    int   height      = 0;
    float iou_score   = 0.0f;
    float obj_score   = 0.0f;
    int   instance_id = -1;
    std::vector<uint8_t> data;  // binary mask (0 or 255)
};

struct sam3_detection {
    sam3_box  box;
    float     score     = 0.0f;
    float     iou_score = 0.0f;
    int       instance_id = -1;
    sam3_mask  mask;
    std::vector<float> sam_token;   // raw SAM decoder output token (for obj_ptr)
    float      obj_logit = 0.0f;    // raw object-score logit (3.1 gating input)
    std::vector<float> mask_logits; // opt-in: pre-binarization logits at mask
                                    // resolution (width x height), filled only
                                    // when return_logits is set; empty otherwise
    std::vector<float> mask_logits_lowres; // opt-in: pre-binarization logits at
                                    // the native decoder resolution (288x288
                                    // for 3.1), filled only when return_logits
                                    // is set; the official memory encoder
                                    // consumes these directly
};

struct sam3_result {
    std::vector<sam3_detection> detections;
};

/*****************************************************************************
** Parameters
**
** Configuration for model loading, segmentation, and video tracking.
*****************************************************************************/

// Backend device selection. AUTO probes the ggml backend registry in order
// (CUDA -> Vulkan -> CPU) and uses the first one that initialises.
// `use_gpu = false` still forces CPU for backwards compatibility.
enum sam3_device {
    SAM3_DEVICE_AUTO   = 0,
    SAM3_DEVICE_CPU    = 1,
    SAM3_DEVICE_CUDA   = 2,
    SAM3_DEVICE_VULKAN = 3,
};

// Explicit diagnostics switches. The library performs zero environment
// lookups; every knob below must be set programmatically through
// sam3_params::debug (replaces the former SAM3_CENSUS / SAM3_PROFILE_PROP /
// SAM3_PCS_PROF / SAM3_ENCODE_TIMING / SAM31_DUMP_BLOCKS / SAM2_DUMP_DIR
// environment variables).
struct sam3_debug_options {
    int         census          = 0;    // 1 = graph/op census, 2 = + CPY shape dump
    int         profile_prop    = 0;    // 1 = propagate + memenc stage timings
    int         pcs_prof        = 0;    // 1 = PCS stage timings
    int         encode_timing   = 0;    // 1 = image-encode graph timing
    int         dump_vit_blocks = 0;    // 1 = dump ViT block tensors (debug builds)
    std::string sam2_dump_dir;          // non-empty = dump SAM2 intermediate tensors here
    std::string parity_dump_dir;        // non-empty = dump mux-path graph tensors (f32)
                                        // for numeric alignment vs official PyTorch
};

struct sam3_params {
    std::string model_path;
    int         n_threads       = 4;
    bool        use_gpu         = true;
    sam3_device device          = SAM3_DEVICE_AUTO;  // explicit device; AUTO picks CUDA->Vulkan->CPU
    int         seed            = 42;
    int         encode_img_size = 0;  // 0 = model default; override input resolution
    sam3_debug_options debug;         // diagnostics; defaults disable all logging
};

struct sam3_tensor_info {
    int64_t ne[4] = {0, 0, 0, 0};
    uint64_t nb[4] = {0, 0, 0, 0};
    int type = 0;
    int op = 0;
    bool is_contiguous = false;
};

enum sam3_vit_block_stage {
    SAM3_VIT_BLOCK_STAGE_NORM1 = 0,
    SAM3_VIT_BLOCK_STAGE_WINDOW_PART,
    SAM3_VIT_BLOCK_STAGE_QKV_PROJ,
    SAM3_VIT_BLOCK_STAGE_ATTN_CORE,
    SAM3_VIT_BLOCK_STAGE_ATTN_PROJ,
    SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART,
    SAM3_VIT_BLOCK_STAGE_NORM2,
    SAM3_VIT_BLOCK_STAGE_MLP_FC1,
    SAM3_VIT_BLOCK_STAGE_MLP_GELU,
    SAM3_VIT_BLOCK_STAGE_MLP_FC2,
    SAM3_VIT_BLOCK_STAGE_MLP,
};

enum sam3_vit_prefix_stage {
    SAM3_VIT_PREFIX_STAGE_PATCH_EMBED = 0,
    SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL,
    SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW,
    SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT,
    SAM3_VIT_PREFIX_STAGE_POS_ADD,
    SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM,
    SAM3_VIT_PREFIX_STAGE_LN_PRE,
};

struct sam3_pcs_params {
    std::string            text_prompt;
    std::vector<sam3_box>  pos_exemplars;
    std::vector<sam3_box>  neg_exemplars;
    float                  score_threshold = 0.5f;
    float                  nms_threshold   = 0.1f;
    bool                   return_logits   = false;  // fill mask_logits per detection
};

struct sam3_pvs_params {
    std::vector<sam3_point> pos_points;
    std::vector<sam3_point> neg_points;
    sam3_box                box      = {0, 0, 0, 0};
    bool                    use_box  = false;
    bool                    multimask = false;
    bool                    return_logits = false;  // fill mask_logits per detection
    /*
    ** Low-resolution mask prompt (SAM 1-task style refinement), matching the
    ** official PromptEncoder mask branch (mask_input_size = 4x the embedding
    ** grid, i.e. 256x256 for a 64x64 embedding; row-major, y*x_width + x).
    ** mask_prompt_logits holds raw continuous logits (typically the previous
    ** prediction's logits, as in the official iterative-refinement flow) and
    ** takes precedence when non-empty. Otherwise the binary convenience mask
    ** (0/255, any resolution) is bilinearly resampled to the same input size.
    ** Either input replaces the no-mask dense embedding; composable with
    ** points and boxes. Note: a mask-only prompt keeps the single trailing
    ** padding token in the sparse stream (the official API emits an empty
    ** sparse stream for that corner case), so mask+point/box combinations are
    ** the exact-alignment path.
    */
    std::vector<float>      mask_prompt_logits;  // [4H x 4W] raw logits
    sam3_mask               mask_prompt;         // binary 0/255 convenience input
    bool                    use_mask_prompt = false;
};

struct sam3_video_params {
    std::string text_prompt;
    float       score_threshold     = 0.5f;
    float       nms_threshold       = 0.1f;
    float       assoc_iou_threshold = 0.1f;
    int         hotstart_delay      = 15;
    int         max_keep_alive      = 30;
    int         recondition_every   = 16;
    int         fill_hole_area      = 16;
    /* Optional total video length hint. The official tracker normalizes the
    ** object-pointer temporal encoding by min(num_frames, 16) - 1; when this
    ** is <= 0 (unknown, e.g. streaming), max_obj_ptrs (16) is assumed, which
    ** is exact for videos of 16+ frames. */
    int         total_frames        = -1;
};

struct sam3_video_info {
    int   width    = 0;
    int   height   = 0;
    int   n_frames = 0;
    float fps      = 0.0f;
};

/*****************************************************************************
** Public API
**
** Model lifecycle, image encoding, segmentation, and video tracking.
*****************************************************************************/

/*
** ── Model Lifecycle ──────────────────────────────────────────────────────
*/

/*
** Load a SAM3 model from the file specified in params.model_path.
** Returns nullptr on failure.
*/
SAM3_API std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params);

/* Free all resources held by a loaded model. */
SAM3_API void sam3_free_model(sam3_model & model);

/* Returns true if the model was loaded as visual-only (no text/detector path).
** SAM2 models are always considered visual-only. */
SAM3_API bool sam3_is_visual_only(const sam3_model & model);

/* Returns the model type (SAM2 or SAM3). */
SAM3_API sam3_model_type sam3_get_model_type(const sam3_model & model);

/* Returns the human-readable name of the active inference backend
** (e.g. "CUDA", "Metal", "Vulkan", "CPU"). Useful for reporting which
** device a model actually runs on. */
SAM3_API const char * sam3_backend_name(const sam3_model & model);

/*
** ── Inference State ──────────────────────────────────────────────────────
*/

/* Allocate inference state (backbone caches, PE buffers). */
SAM3_API sam3_state_ptr sam3_create_state(const sam3_model & model,
                                const sam3_params & params);

/* Free inference state and its GPU buffers. */
SAM3_API void sam3_free_state(sam3_state & state);

/*
** ── Image Backbone ───────────────────────────────────────────────────────
*/

/*
** Encode an image through the ViT backbone and FPN neck.
** Call once per image before segmentation or tracking.
** Returns true on success, false on failure.
*/
SAM3_API bool sam3_encode_image(sam3_state       & state,
                       const sam3_model & model,
                       const sam3_image & image);

/*
** Encode only the image features consumed by point/box segmentation and
** tracking. For full SAM3 checkpoints this skips the detector neck; use
** sam3_encode_image() when sam3_segment_pcs() will consume the image.
*/
SAM3_API bool sam3_encode_image_pvs(sam3_state       & state,
                           const sam3_model & model,
                           const sam3_image & image);

/*
** ── Image Segmentation ──────────────────────────────────────────────────
*/

/* Segment using text prompt + exemplar boxes (PCS path). */
SAM3_API sam3_result sam3_segment_pcs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pcs_params  & params);

/*
** Batch image inference (official Sam3Processor::set_image_batch flow).
**
** sam3_encode_image_batch encodes n images and keeps a per-image snapshot of
** the detector neck features inside the state (a new call replaces the
** previous batch). sam3_segment_pcs_batch then applies one text prompt to
** every image of that batch and writes one sam3_result per image (masks at
** each image's original resolution). The detector attends within each image
** only, so batch results are identical to encode-and-query-each separately.
**
** Returns: number of images processed, or -1 on failure (also when the model
** has no detector path — SAM2 / visual-only checkpoints).
*/
SAM3_API int sam3_encode_image_batch(sam3_state       & state,
                            const sam3_model & model,
                            const sam3_image * images,
                            int                n);
SAM3_API int sam3_segment_pcs_batch(sam3_state             & state,
                           const sam3_model       & model,
                           const sam3_pcs_params  & params,
                           sam3_result            * results,
                           int                      n);

/* Segment using point/box prompts (PVS path). */
SAM3_API sam3_result sam3_segment_pvs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pvs_params  & params);

/* Cross-DLL allocator pairing: drain a sam3_result's vectors inside the DLL
** that allocated them. When sam3 is built as a shared library and the caller
** lives in a different module (e.g. an Unreal Engine plugin that overrides
** global ::operator new/delete), letting ~sam3_result fire on the caller
** side inlines std::vector<>::~vector() into the caller's TU and routes
** ::operator delete[] through the caller's allocator — which doesn't
** recognize blocks created here. Symptom: a heap-canary fatal in the host
** allocator (e.g. UE's FMallocBinned2 "Attempt to realloc an unrecognized
** block"). Call this immediately after extracting data; the result is left
** valid-but-empty so the caller's eventual ~sam3_result is a no-op. */
SAM3_API void sam3_free_result(sam3_result & result) noexcept;

/*
** ── Video Tracking ──────────────────────────────────────────────────────
*/

/* Create a tracker for text-prompted video segmentation. */
SAM3_API sam3_tracker_ptr sam3_create_tracker(const sam3_model       & model,
                                    const sam3_video_params & params);

/* Encode a frame, detect objects, and update tracked instances. */
SAM3_API sam3_result sam3_track_frame(sam3_tracker     & tracker,
                             sam3_state       & state,
                             const sam3_model & model,
                             const sam3_image & frame);

/*
** Refine a tracked instance with interactive point prompts. The image of the
** refined frame must already be encoded. frame_idx selects the frame the
** refinement applies to: -1 (default) = the frame just processed
** (tracker.frame_index - 1); an explicit value must be < tracker.frame_index.
** After the call, tracker.frame_index points at the refined frame, so the
** next forward propagate re-processes it first (official
** propagate_in_video(start_frame_idx) semantics).
*/
SAM3_API bool sam3_refine_instance(sam3_tracker                   & tracker,
                          sam3_state                     & state,
                          const sam3_model               & model,
                          int                              instance_id,
                          const std::vector<sam3_point>  & pos_points,
                          const std::vector<sam3_point>  & neg_points,
                          int                              frame_idx = -1);

/*
** Add a new instance to the tracker from PVS prompts (points/box).  The image
** of the prompt frame must already be encoded (via sam3_track_frame,
** sam3_encode_image, or sam3_propagate_frame).  frame_idx selects the
** conditioning frame: -1 (default) = the frame just processed
** (tracker.frame_index - 1, or frame 0 on a fresh tracker); an explicit
** value conditions on that frame (the caller must have encoded its image)
** and positions the tracker there — forward propagation then starts by
** re-processing the prompt frame (official range(start, end]) and reverse
** propagation starts at frame_idx-1 (official range(start-1, ..., -1)).
** Returns assigned instance_id, or -1 on failure.
*/
SAM3_API int sam3_tracker_add_instance(sam3_tracker         & tracker,
                              sam3_state            & state,
                              const sam3_model      & model,
                              const sam3_pvs_params & pvs_params,
                              int                     frame_idx = -1);

/*
** Add a new instance to the tracker from an existing binary mask, bypassing
** the PVS prompt encoder / mask decoder.  The image of the prompt frame must
** already be encoded (via sam3_track_frame, sam3_propagate_frame, or
** sam3_encode_image).  The mask (0/255, any resolution) is resampled to the
** tracker's memory resolution and written straight into the memory bank as a
** conditioning frame, so propagation tracks exactly the supplied mask rather
** than a mask re-derived from points/box.  frame_idx follows the same
** convention as sam3_tracker_add_instance.
**
** The prompt path produces no SAM decoder token, so the object pointer is
** instead obtained by running one propagation decode against the just-written
** conditioning memory (the same SAM token every tracked frame yields) and
** projecting it; this gives propagation a real appearance cue to re-localize
** with.  obj_score is the confidence stored for the seed frame (1.0 = fully
** trusted user mask).
**
** Returns the assigned instance_id, or -1 on failure (empty mask).
*/
SAM3_API int sam3_tracker_add_instance_from_mask(sam3_tracker     & tracker,
                                        sam3_state       & state,
                                        const sam3_model & model,
                                        const sam3_mask  & mask,
                                        float              obj_score = 1.0f,
                                        int                frame_idx = -1);

/* Return the current frame index of the tracker. */
SAM3_API int  sam3_tracker_frame_index(const sam3_tracker & tracker);

/* Reset the tracker, clearing all instances and memory. */
SAM3_API void sam3_tracker_reset(sam3_tracker & tracker);

/*
** Remove a tracked instance by ID (aligns with the official video predictor's
** remove_object session edit): drops its confirmed masklet, pending hotstart
** entry, memory bank and object-pointer bank. The instance's tensor buffers
** are reclaimed together with the tracker's shared allocations on reset,
** matching the sliding-window eviction policy used by the memory banks.
** Returns true if the instance existed.
*/
SAM3_API bool sam3_tracker_remove_instance(sam3_tracker & tracker, int instance_id);

/*
** Rewind the tracker to an earlier frame so a prompt can be added there
** (aligns with the official add_prompt-at-earlier-frame +
** propagate_in_video(start_frame_idx=...) flow): drops every memory slot and
** object pointer recorded after `frame_index`, removes instances that first
** appeared after it (they do not exist at that frame yet), and repositions
** the tracker so a following sam3_tracker_add_instance /
** sam3_refine_instance (default frame_idx = -1) lands exactly on
** `frame_index`, and the next forward propagate re-processes that frame
** first, exactly like the official replay flow. Returns the number of
** dropped memory slots, or -1 if the frame index is invalid.
*/
SAM3_API int sam3_tracker_rewind(sam3_tracker & tracker, int frame_index);

/*
** Clear the prompt/conditioning entry of one tracked instance at one frame
** (aligns with the official clear_all_points_in_frame): drops that frame's
** memory slot for the instance, so subsequent propagation is no longer
** constrained by it. To rebuild the track, rewind to that frame, re-prompt
** via sam3_refine_instance, and replay. Returns true if a slot was removed.
*/
SAM3_API bool sam3_tracker_clear_instance_frame(sam3_tracker & tracker, int instance_id, int frame_index);

/*
** ── Visual-Only Video Tracking ──────────────────────────────────────────
*/

struct sam3_visual_track_params {
    float assoc_iou_threshold = 0.1f;
    int   max_keep_alive      = 30;
    int   recondition_every   = 16;
    int   fill_hole_area      = 16;
    /* Optional total video length hint for the object-pointer temporal
    ** encoding (see sam3_video_params::total_frames). */
    int   total_frames        = -1;
};

/*
** Create a tracker for visual-only models.  Instances are added manually
** via sam3_tracker_add_instance().
*/
SAM3_API sam3_tracker_ptr sam3_create_visual_tracker(
    const sam3_model               & model,
    const sam3_visual_track_params & params);

/*
** Propagate all tracked instances to the next frame (no detection step).
** The image is encoded, then each tracked instance is propagated via
** memory attention + SAM mask decode, and the memory bank is updated.
**
** The passed frame must be the next *unprocessed* frame in the tracking
** direction: after add_instance on frame f, pass frame f+1 (forward) or
** f-1 (reverse). Frames whose outputs are already consolidated (the
** annotation frame, and frames processed by an earlier pass) are skipped
** exactly like the official propagate_in_video
** (consolidated_frame_inds / frames_already_tracked check): their outputs
** already exist and their memory is already encoded, so they are never
** re-processed.
**
** reverse=true: aligned with the official propagate_in_video(reverse=True).
** Memory slots and object pointers on the future side of the current frame
** participate, with temporal positions folded to non-negative distances in
** tracking order (official tpos_sign_mul). Hotstart/keep-alive are measured
** in tracking order.
*/
SAM3_API sam3_result sam3_propagate_frame(
    sam3_tracker     & tracker,
    sam3_state       & state,
    const sam3_model & model,
    const sam3_image & frame,
    bool               reverse = false);

/*
** ── Utility ─────────────────────────────────────────────────────────────
*/

SAM3_API sam3_image      sam3_load_image(const std::string & path);
SAM3_API bool            sam3_save_mask(const sam3_mask & mask, const std::string & path);
SAM3_API sam3_image      sam3_decode_video_frame(const std::string & video_path, int frame_index);
SAM3_API sam3_video_info sam3_get_video_info(const std::string & video_path);

/*
** ── Visualization ───────────────────────────────────────────────────────
**
** Zero-dependency rendering of segmentation results, aligned in content
** with the official repo's visualization utilities (matplotlib-based):
** per-instance colored mask overlay + box outlines + "id score" labels.
** Implemented with plain pixel blending + a built-in 5x7 dot-matrix font.
*/

struct sam3_vis_params {
    float mask_alpha    = 0.35f;  // mask tint opacity in [0,1]
    bool  draw_boxes    = true;   // draw detection box outlines
    bool  draw_labels   = true;   // draw "id score" labels above boxes
    int   box_thickness = 2;      // box outline thickness in pixels
    int   font_scale    = 2;      // label glyph scale (base glyph is 5x7)
};

/* Fixed perceptually-balanced instance palette (pre-generated table; the
** official repo clusters LAB samples with k-means, which is a static
** resource rather than a runtime feature — hard-coding removes the runtime
** clustering entirely). instance_id 1 maps to the first color. */
SAM3_API void sam3_instance_color(int instance_id, uint8_t & r, uint8_t & g, uint8_t & b);

/* Blend one binary mask (0/255, any resolution — nearest-neighbour resampled
** when its size differs from the image) onto the image with the given color. */
SAM3_API void sam3_overlay_mask(sam3_image  & image,
                       const sam3_mask & mask,
                       uint8_t r, uint8_t g, uint8_t b,
                       float alpha);

/* Draw an axis-aligned box outline (clipped to image bounds). */
SAM3_API void sam3_draw_box(sam3_image & image, const sam3_box & box,
                   uint8_t r, uint8_t g, uint8_t b, int thickness);

/* Full-scene render: copy of `image` with every detection's mask blended in
** its instance color, then boxes and "id score" labels. This is the C++
** counterpart of the official full-scene instance visualization. */
SAM3_API sam3_image sam3_render_result(const sam3_image      & image,
                              const sam3_result     & result,
                              const sam3_vis_params & params);

/* Save an RGB image as PNG. */
SAM3_API bool sam3_save_image(const sam3_image & image, const std::string & path);

/*
** Bilinear-resize raw mask logits (e.g. a previous prediction's
** detection.mask_logits, or any logits grid) to the low-resolution mask-prompt
** input size (4x the embedding grid, i.e. 256x256 for the shipped models).
** dst must hold dw*dh floats. This is the companion helper for
** sam3_pvs_params::mask_prompt_logits, mirroring the official flow where the
** previous prediction's logits feed the next prompt round.
*/
SAM3_API void sam3_resize_mask_logits(const float * src, int sw, int sh,
                             float * dst, int dw, int dh);

/*
** Greedy IoM (intersection-over-min-area) mask dedup, matching the official
** agent helper remove_overlapping_masks(): process detections by score
** descending, keep a detection only if its mask IoM against every already
** kept mask is <= iom_threshold (official default 0.3), then restore the
** kept detections in their original relative order. Returns the number of
** removed detections.
*/
SAM3_API int sam3_remove_overlapping_masks(sam3_result & result, float iom_threshold);

/*****************************************************************************
** Test and Debug API
**
** Standalone tokenizer, intermediate tensor dumps, and debug utilities.
** These functions are intended for testing and development only.
*****************************************************************************/

SAM3_API bool                  sam3_test_load_tokenizer(const std::string & model_path);
SAM3_API std::vector<int32_t>  sam3_test_tokenize(const std::string & text);
/* Expose the official clean chain (whitespace_clean(basic_clean(t)).lower())
** for differential testing against the Python tokenizer. */
SAM3_API std::string           sam3_test_clean_text(const std::string & text);

/*
** Run the text encoder on fixed token IDs and dump standard intermediate
** tensors to <output_dir>/<tensor_name>.{bin,shape}.
*/
SAM3_API bool sam3_test_dump_text_encoder(const sam3_model & model,
                                 const std::vector<int32_t> & token_ids,
                                 const std::string & output_dir,
                                 int n_threads = 4);

/*
** Run the full phase 5 detector path (fusion encoder + DETR decoder +
** dot-product scoring + segmentation head) on an already-encoded image
** and dump intermediate tensors.
*/
SAM3_API bool sam3_test_dump_phase5(const sam3_model & model,
                           const sam3_state & state,
                           const std::vector<int32_t> & token_ids,
                           const std::string & output_dir,
                           int n_threads = 4);

/*
** Run the phase 5 detector from pre-dumped inputs instead of re-running
** the image/text encoders.  Isolates detector numerics from earlier phases.
*/
SAM3_API bool sam3_test_dump_phase5_from_ref_inputs(const sam3_model & model,
                                           const std::vector<int32_t> & token_ids,
                                           const std::string & prephase_ref_dir,
                                           const std::string & phase5_ref_dir,
                                           const std::string & output_dir,
                                           int n_threads = 4);

/*
** Run the phase 6 prompt encoder + SAM decoder on an already-encoded
** tracker image state and dump intermediate tensors.
*/
SAM3_API bool sam3_test_dump_phase6(const sam3_model & model,
                           const sam3_state & state,
                           const sam3_pvs_params & params,
                           const std::string & output_dir,
                           int n_threads = 4);

/*
** Run the phase 6 prompt encoder + SAM decoder from pre-dumped phase 3
** tracker features.  Isolates phase 6 numerics from earlier phases.
*/
SAM3_API bool sam3_test_dump_phase6_from_ref_inputs(const sam3_model & model,
                                           const std::string & prephase_ref_dir,
                                           const sam3_pvs_params & params,
                                           const std::string & output_dir,
                                           int n_threads = 4);

/*
** Run the phase 7 tracker subgraph from pre-dumped case inputs and dump
** intermediate tensors.  Case directory produced by dump_phase7_reference.py.
*/
SAM3_API bool sam3_test_dump_phase7_from_ref_inputs(const sam3_model & model,
                                           const std::string & case_ref_dir,
                                           const std::string & output_dir,
                                           int n_threads = 4);

/*
** Run the geometry encoder from pre-computed backbone features and dump
** intermediate tensors.  Tests exemplar box coordinate encoding against
** Python reference.
*/
SAM3_API bool sam3_test_dump_geom_enc(const sam3_model   & model,
                              const std::string  & prephase_ref_dir,
                              const sam3_pcs_params & params,
                              const std::string  & output_dir,
                              int                  n_threads = 4);

/*
** Run ONLY the fusion encoder (6 layers) from pre-dumped inputs (image
** features, pos encoding, prompt tokens, attn bias).  Dumps per-layer
** outputs for isolated fenc debugging.
*/
SAM3_API bool sam3_test_fenc_only(const sam3_model  & model,
                          const std::string & ref_dir,
                          const std::string & output_dir,
                          int                 n_threads = 4);

/*
** ── Debug ────────────────────────────────────────────────────────────────
*/

/* Dump a named state tensor to a binary file for verification. */
SAM3_API bool sam3_dump_state_tensor(const sam3_state & state,
                             const std::string & tensor_name,
                             const std::string & output_path);

/* Query metadata for a named state tensor without dumping its payload. */
SAM3_API bool sam3_get_state_tensor_info(const sam3_state & state,
                                const std::string & tensor_name,
                                sam3_tensor_info  & info);

/* Dump a named model tensor (weights/constants) to a binary file. */
SAM3_API bool sam3_dump_model_tensor(const sam3_model   & model,
                            const std::string  & tensor_name,
                            const std::string  & output_path);

/* Query metadata for a named model tensor. */
SAM3_API bool sam3_get_model_tensor_info(const sam3_model  & model,
                                const std::string & tensor_name,
                                sam3_tensor_info  & info);

/*
** Encode an image from pre-preprocessed float data (CHW layout, already
** resized and normalized).  Bypasses C++ preprocessing so that numerical
** comparisons against the Python reference are not polluted by resize
** implementation differences.
*/
SAM3_API bool sam3_encode_image_from_preprocessed(sam3_state       & state,
                                          const sam3_model & model,
                                          const float      * chw_data,
                                          int                img_size);

/*
** Test-only: run ONLY the ViT encoder from preprocessed float data and keep
** only the requested intermediate tensors alive for dumping/comparison.
*/
SAM3_API bool sam3_encode_vit_from_preprocessed_selective(sam3_state                    & state,
                                                 const sam3_model              & model,
                                                 const float                   * chw_data,
                                                 int                             img_size,
                                                 const std::vector<std::string> & output_tensors);

/*
** Test-only: run the exact ViT prefix up to the tensor entering block 0
** (patch embed + pos embed + ln_pre).
*/
SAM3_API bool sam3_test_run_vit_block0_input(const sam3_model   & model,
                                    const float        * chw_data,
                                    int                  img_size,
                                    std::vector<float> & output_data,
                                    int64_t              output_ne[4],
                                    int                  n_threads = 4);

/*
** Test-only: run an exact ViT prefix sub-stage on the real model tensors using
** the model's backend. PATCH_EMBED expects image input [W,H,3,1]. Later stages
** expect feature input [E,W,H,1] in ggml 4D layout.
*/
SAM3_API bool sam3_test_run_vit_prefix_stage(const sam3_model         & model,
                                    sam3_vit_prefix_stage      stage,
                                    const float              * input_data,
                                    const int64_t              input_ne[4],
                                    std::vector<float>       & output_data,
                                    int64_t                    output_ne[4],
                                    int                        n_threads = 4);
SAM3_API bool sam3_test_run_patch_mulmat_host_ref(const sam3_model         & model,
                                         const float              * input_data,
                                         const int64_t              input_ne[4],
                                         bool                       use_double_accum,
                                         std::vector<float>       & output_data,
                                         int64_t                    output_ne[4]);
SAM3_API bool sam3_test_run_vit_block_linear_host_ref(const sam3_model         & model,
                                             int                        block_idx,
                                             sam3_vit_block_stage       stage,
                                             const float              * input_data,
                                             const int64_t              input_ne[4],
                                             bool                       use_double_accum,
                                             std::vector<float>       & output_data,
                                             int64_t                    output_ne[4]);

/*
** Test-only: run an exact ViT block sub-stage on the real model tensors using
** the model's backend. The input is always provided as F32 in ggml 4D layout.
*/
SAM3_API bool sam3_test_run_vit_block_stage(const sam3_model        & model,
                                   int                       block_idx,
                                   sam3_vit_block_stage      stage,
                                   const float             * input_data,
                                   const int64_t             input_ne[4],
                                   std::vector<float>      & output_data,
                                   int64_t                   output_ne[4],
                                   int                       n_threads = 4);
