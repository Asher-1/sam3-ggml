// Shared CLI plumbing for the diagnostic knobs of sam3_debug_options.
//
// The library performs zero environment lookups; examples surface those
// knobs as explicit command-line flags through this helper. Drop one
// `else if (sam3_example_debug_flag(argv[i], params.debug)) { }` branch
// into the example's argument loop.
#pragma once
#include <cstring>
#include <string>
#include "sam3.h"

// Returns true if `arg` was recognized as a debug flag (and consumed).
inline bool sam3_example_debug_flag(const char* arg, sam3_debug_options& dbg) {
    if (std::strcmp(arg, "--census") == 0)              { dbg.census = 1;          return true; }
    if (std::strcmp(arg, "--census=2") == 0)            { dbg.census = 2;          return true; }
    if (std::strcmp(arg, "--profile-prop") == 0)        { dbg.profile_prop = 1;    return true; }
    if (std::strcmp(arg, "--pcs-prof") == 0)            { dbg.pcs_prof = 1;        return true; }
    if (std::strcmp(arg, "--encode-timing") == 0)       { dbg.encode_timing = 1;   return true; }
    if (std::strcmp(arg, "--dump-vit-blocks") == 0)     { dbg.dump_vit_blocks = 1; return true; }
    if (std::strncmp(arg, "--sam2-dump-dir=", 16) == 0) { dbg.sam2_dump_dir = arg + 16; return true; }
    return false;
}

// One-line usage text for the flags above (append to example help).
inline const char* sam3_example_debug_usage() {
    return "  --census / --census=2    graph/op census (optionally dump CPY shapes)\n"
           "  --profile-prop           propagate + memenc stage timings\n"
           "  --pcs-prof               PCS stage timings\n"
           "  --encode-timing          image-encode graph timing\n"
           "  --dump-vit-blocks        dump ViT block tensors\n"
           "  --sam2-dump-dir=DIR      dump SAM2 intermediate tensors to DIR\n";
}
