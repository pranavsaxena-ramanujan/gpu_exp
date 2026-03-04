#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

// Vendored Khronos OpenCL headers (no NDK cl.h; we dlopen at runtime)
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"

// Dynamic OpenCL loader — resolves all cl* symbols via dlopen/dlsym
#include "opencl/opencl_loader.h"

#define LOG_TAG "MatMulCL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Helper: check OpenCL error and return empty array on failure
static bool checkCL(cl_int err, const char* op, JNIEnv* env, jfloatArray& out) {
    if (err != CL_SUCCESS) {
        LOGE("%s failed with error %d", op, err);
        out = env->NewFloatArray(0);
        return false;
    }
    return true;
}

// ── Native CPU matrix multiply ──────────────────────────────────────────────
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_in_ramanujan_gpuexp_MainActivity_matMulCPU(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray aArr,
        jfloatArray bArr,
        jint n)
{
    jsize len = n * n;
    std::vector<float> hA(len), hB(len), hC(len, 0.0f);
    env->GetFloatArrayRegion(aArr, 0, len, hA.data());
    env->GetFloatArrayRegion(bArr, 0, len, hB.data());

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += hA[row * n + k] * hB[k * n + col];
            }
            hC[row * n + col] = sum;
        }
    }

    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, hC.data());
    return result;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_in_ramanujan_gpuexp_MainActivity_matMulCL(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray aArr,
        jfloatArray bArr,
        jint n,
        jstring kernelSrcJ)
{
    jfloatArray result = nullptr;
    cl_int err = CL_SUCCESS;

    // ── 0. Load OpenCL at runtime ───────────────────────────────────────────
    if (!openclLoad()) {
        LOGE("OpenCL runtime not available on this device");
        return env->NewFloatArray(0);
    }

    // ── 1. Convert Java arrays ──────────────────────────────────────────────
    jsize len = n * n;
    std::vector<float> hA(len), hB(len), hC(len, 0.0f);
    env->GetFloatArrayRegion(aArr, 0, len, hA.data());
    env->GetFloatArrayRegion(bArr, 0, len, hB.data());

    const char* kernelSrc = env->GetStringUTFChars(kernelSrcJ, nullptr);

    // ── 2. Platform / Device ────────────────────────────────────────────────
    cl_platform_id platform = nullptr;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        LOGE("clGetPlatformIDs failed (%d) or no platforms found", err);
        env->ReleaseStringUTFChars(kernelSrcJ, kernelSrc);
        return env->NewFloatArray(0);
    }

    cl_device_id device = nullptr;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        LOGI("GPU not found (%d), falling back to CPU", err);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            LOGE("No OpenCL device available (%d)", err);
            env->ReleaseStringUTFChars(kernelSrcJ, kernelSrc);
            return env->NewFloatArray(0);
        }
    }

    // ── 3. Context & Queue ──────────────────────────────────────────────────
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!checkCL(err, "clCreateContext", env, result)) {
        env->ReleaseStringUTFChars(kernelSrcJ, kernelSrc);
        return result;
    }

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (!checkCL(err, "clCreateCommandQueue", env, result)) {
        clReleaseContext(ctx);
        env->ReleaseStringUTFChars(kernelSrcJ, kernelSrc);
        return result;
    }

    // ── 4. Buffers ──────────────────────────────────────────────────────────
    size_t bytes = sizeof(float) * len;
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, bytes, hA.data(), &err);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, bytes, hB.data(), &err);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,                          bytes, nullptr,  &err);

    // ── 5. Build program ────────────────────────────────────────────────────
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, &err);
    if (!checkCL(err, "clCreateProgramWithSource", env, result)) goto cleanup;

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Dump build log
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        LOGE("Build log:\n%s", log.c_str());
        result = env->NewFloatArray(0);
        goto cleanup;
    }

    {
        // ── 6. Kernel & args ────────────────────────────────────────────────
        cl_kernel kernel = clCreateKernel(program, "matMul", &err);
        if (!checkCL(err, "clCreateKernel", env, result)) goto cleanup;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(cl_int), &n);

        // ── 7. Enqueue ──────────────────────────────────────────────────────
        size_t globalSize[2] = { (size_t)n, (size_t)n };
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
        if (!checkCL(err, "clEnqueueNDRangeKernel", env, result)) {
            clReleaseKernel(kernel);
            goto cleanup;
        }

        // ── 8. Read back ────────────────────────────────────────────────────
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, hC.data(), 0, nullptr, nullptr);
        clReleaseKernel(kernel);
        if (!checkCL(err, "clEnqueueReadBuffer", env, result)) goto cleanup;

        // ── 9. Pack result ──────────────────────────────────────────────────
        result = env->NewFloatArray(len);
        env->SetFloatArrayRegion(result, 0, len, hC.data());
    }

cleanup:
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    if (program) clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    env->ReleaseStringUTFChars(kernelSrcJ, kernelSrc);
    return result;
}
