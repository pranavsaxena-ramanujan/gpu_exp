#pragma once
// Thin runtime loader for libOpenCL.so on Android.
// Callers must #include "CL/cl.h" (with CL_TARGET_OPENCL_VERSION=120 defined)
// BEFORE including this header. This file resolves all cl* function pointers
// via dlopen/dlsym and re-#defines each cl* name to its pfn_* pointer so the
// rest of the code uses the normal OpenCL API without any changes.

#include <dlfcn.h>
#include <android/log.h>

#define OPENCL_LOG_TAG "OpenCLLoader"

// CL/cl.h must already be included by the caller — we only need the types.

// ── storage for resolved function pointers ──────────────────────────────────
typedef cl_int          (*PFN_clGetPlatformIDs)          (cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int          (*PFN_clGetDeviceIDs)             (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context      (*PFN_clCreateContext)            (const cl_context_properties*, cl_uint, const cl_device_id*, void(*)(const char*,const void*,size_t,void*), void*, cl_int*);
typedef cl_command_queue(*PFN_clCreateCommandQueue)       (cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_mem          (*PFN_clCreateBuffer)             (cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_program      (*PFN_clCreateProgramWithSource)  (cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int          (*PFN_clBuildProgram)             (cl_program, cl_uint, const cl_device_id*, const char*, void(*)(cl_program,void*), void*);
typedef cl_int          (*PFN_clGetProgramBuildInfo)      (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel       (*PFN_clCreateKernel)             (cl_program, const char*, cl_int*);
typedef cl_int          (*PFN_clSetKernelArg)             (cl_kernel, cl_uint, size_t, const void*);
typedef cl_int          (*PFN_clEnqueueNDRangeKernel)     (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int          (*PFN_clEnqueueReadBuffer)        (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int          (*PFN_clReleaseMemObject)         (cl_mem);
typedef cl_int          (*PFN_clReleaseProgram)           (cl_program);
typedef cl_int          (*PFN_clReleaseKernel)            (cl_kernel);
typedef cl_int          (*PFN_clReleaseCommandQueue)      (cl_command_queue);
typedef cl_int          (*PFN_clReleaseContext)           (cl_context);
typedef cl_int          (*PFN_clFinish)                   (cl_command_queue);

static void* g_clLib = nullptr;

static PFN_clGetPlatformIDs         pfn_clGetPlatformIDs         = nullptr;
static PFN_clGetDeviceIDs           pfn_clGetDeviceIDs           = nullptr;
static PFN_clCreateContext          pfn_clCreateContext          = nullptr;
static PFN_clCreateCommandQueue     pfn_clCreateCommandQueue     = nullptr;
static PFN_clCreateBuffer           pfn_clCreateBuffer           = nullptr;
static PFN_clCreateProgramWithSource pfn_clCreateProgramWithSource = nullptr;
static PFN_clBuildProgram           pfn_clBuildProgram           = nullptr;
static PFN_clGetProgramBuildInfo    pfn_clGetProgramBuildInfo    = nullptr;
static PFN_clCreateKernel           pfn_clCreateKernel           = nullptr;
static PFN_clSetKernelArg           pfn_clSetKernelArg           = nullptr;
static PFN_clEnqueueNDRangeKernel   pfn_clEnqueueNDRangeKernel   = nullptr;
static PFN_clEnqueueReadBuffer      pfn_clEnqueueReadBuffer      = nullptr;
static PFN_clReleaseMemObject       pfn_clReleaseMemObject       = nullptr;
static PFN_clReleaseProgram         pfn_clReleaseProgram         = nullptr;
static PFN_clReleaseKernel          pfn_clReleaseKernel          = nullptr;
static PFN_clReleaseCommandQueue    pfn_clReleaseCommandQueue    = nullptr;
static PFN_clReleaseContext         pfn_clReleaseContext         = nullptr;
static PFN_clFinish                 pfn_clFinish                 = nullptr;

#define LOAD_SYM(lib, name) \
    pfn_##name = reinterpret_cast<PFN_##name>(dlsym(lib, #name)); \
    if (!pfn_##name) { \
        __android_log_print(ANDROID_LOG_ERROR, OPENCL_LOG_TAG, "dlsym failed for " #name); \
        return false; \
    }

// Returns true if libOpenCL.so was loaded and all symbols resolved.
static bool openclLoad() {
    if (g_clLib) return true; // already loaded

    // The manifest declares <uses-native-library android:name="libOpenCL.so">
    // which grants our linker namespace access to the vendor library.
    // RTLD_GLOBAL lets subsequently loaded libraries also find the symbols.
    g_clLib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_clLib) {
        __android_log_print(ANDROID_LOG_ERROR, OPENCL_LOG_TAG,
                            "Could not load libOpenCL.so: %s", dlerror());
        return false;
    }

    LOAD_SYM(g_clLib, clGetPlatformIDs)
    LOAD_SYM(g_clLib, clGetDeviceIDs)
    LOAD_SYM(g_clLib, clCreateContext)
    LOAD_SYM(g_clLib, clCreateCommandQueue)
    LOAD_SYM(g_clLib, clCreateBuffer)
    LOAD_SYM(g_clLib, clCreateProgramWithSource)
    LOAD_SYM(g_clLib, clBuildProgram)
    LOAD_SYM(g_clLib, clGetProgramBuildInfo)
    LOAD_SYM(g_clLib, clCreateKernel)
    LOAD_SYM(g_clLib, clSetKernelArg)
    LOAD_SYM(g_clLib, clEnqueueNDRangeKernel)
    LOAD_SYM(g_clLib, clEnqueueReadBuffer)
    LOAD_SYM(g_clLib, clReleaseMemObject)
    LOAD_SYM(g_clLib, clReleaseProgram)
    LOAD_SYM(g_clLib, clReleaseKernel)
    LOAD_SYM(g_clLib, clReleaseCommandQueue)
    LOAD_SYM(g_clLib, clReleaseContext)
    LOAD_SYM(g_clLib, clFinish)

    return true;
}

// Redirect bare CL calls to the loaded function pointers
#define clGetPlatformIDs          pfn_clGetPlatformIDs
#define clGetDeviceIDs            pfn_clGetDeviceIDs
#define clCreateContext           pfn_clCreateContext
#define clCreateCommandQueue      pfn_clCreateCommandQueue
#define clCreateBuffer            pfn_clCreateBuffer
#define clCreateProgramWithSource pfn_clCreateProgramWithSource
#define clBuildProgram            pfn_clBuildProgram
#define clGetProgramBuildInfo     pfn_clGetProgramBuildInfo
#define clCreateKernel            pfn_clCreateKernel
#define clSetKernelArg            pfn_clSetKernelArg
#define clEnqueueNDRangeKernel    pfn_clEnqueueNDRangeKernel
#define clEnqueueReadBuffer       pfn_clEnqueueReadBuffer
#define clReleaseMemObject        pfn_clReleaseMemObject
#define clReleaseProgram          pfn_clReleaseProgram
#define clReleaseKernel           pfn_clReleaseKernel
#define clReleaseCommandQueue     pfn_clReleaseCommandQueue
#define clReleaseContext          pfn_clReleaseContext
#define clFinish                  pfn_clFinish
