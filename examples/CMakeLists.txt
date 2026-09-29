# Example executables for sam3.cpp

# Quantization tool (no GUI dependency)
add_executable(sam3_quantize quantize.cpp)
target_link_libraries(sam3_quantize PRIVATE ggml)

add_executable(sam3_benchmark benchmark.cpp)
target_link_libraries(sam3_benchmark PRIVATE sam3)

# Offline visualization tool (no GUI dependency): renders segmentation
# results (image) or tracked frames (video) to PNG files.
add_executable(sam3_vis main_vis.cpp)
target_link_libraries(sam3_vis PRIVATE sam3)

add_executable(sam3_profile_sam3 profile_sam3.cpp)
target_link_libraries(sam3_profile_sam3 PRIVATE sam3)

# Multi-instance tracking benchmark (mux bucket vs per-instance scaling).
add_executable(sam3_multi_bench multi_inst_bench.cpp)
target_link_libraries(sam3_multi_bench PRIVATE sam3)

# SDL2 + ImGui are optional — build only if SDL2 is found.
find_package(SDL2 QUIET)

if(SDL2_FOUND)
    add_subdirectory(third-party)

    add_executable(sam3_image main_image.cpp)
    target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

    add_executable(sam3_video main_video.cpp)
    target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
else()
    message(STATUS "SDL2 not found — skipping GUI examples")
endif()
