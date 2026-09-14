// CUDA 错误检查与工具宏（整个工程统一使用）
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA runtime API 调用检查：失败即打印位置并终止
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s -> %s\n",                 \
                   __FILE__, __LINE__, #call, cudaGetErrorString(err__));    \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// 内核启动后检查（捕获 launch 配置错误与异步错误）
#define CUDA_CHECK_LAST()                                                    \
  do {                                                                       \
    cudaError_t err__ = cudaGetLastError();                                  \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "[CUDA KERNEL ERROR] %s:%d  %s\n",                \
                   __FILE__, __LINE__, cudaGetErrorString(err__));           \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// 可恢复场景：失败返回错误码而不终止（用于探测类代码）
inline cudaError_t cuda_try(cudaError_t err) { return err; }