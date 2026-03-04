cmake_minimum_required(VERSION 3.22.1)
project("gpuexp")

add_library(
        gpuexp
        SHARED
        matmul_jni.cpp
)

# Vendored Khronos OpenCL headers live in cpp/CL/ — add cpp/ as include root
# so both #include "CL/cl.h" (from loader) and #include "opencl/opencl_loader.h"
# resolve correctly. No NDK cl.h; we use dlopen at runtime.
target_include_directories(gpuexp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find the Android log library
find_library(log-lib log)

# Link against log and dl (needed for dlopen / dlsym in opencl_loader.h).
# No -lOpenCL — the library is loaded dynamically at runtime.
target_link_libraries(
        gpuexp
        ${log-lib}
        dl
)

target_compile_definitions(gpuexp PRIVATE CL_TARGET_OPENCL_VERSION=120)
set_target_properties(gpuexp PROPERTIES CXX_STANDARD 17)
