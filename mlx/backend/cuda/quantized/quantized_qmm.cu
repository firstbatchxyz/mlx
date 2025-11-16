// Copyright © 2025 Apple Inc.

#include <cuda_runtime.h>
#include <cstdint>
#include "mlx/backend/cuda/quantized/quantized_qmm.cuh"

namespace mlx::core {

// Simple CUDA kernel for quantized matmul (affine, non-batched, non-transposed)
__global__ void quantized_qmm_kernel(
    const float* x, // input matrix (dequantized)
    const uint8_t* wq, // quantized weights
    const float* scales, // per-group scales
    const float* biases, // optional
    float* out,
    int M,
    int N,
    int K,
    int group_size,
    int bits) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // TODO: Dequantize wq[k, col] using scales and bits
      float w = 0.0f; // Placeholder: real dequant logic needed
      acc += x[row * K + k] * w;
    }
    if (biases)
      acc += biases[col];
    out[row * N + col] = acc;
  }
}

void quantized_qmm_cuda(
    const array& x,
    const array& wq,
    const array& scales,
    const array* biases,
    array& out,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  int M = x.shape(-2);
  int N = wq.shape(-1);
  int K = x.shape(-1);
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  quantized_qmm_kernel<<<grid, block, 0, enc.stream()>>>(
      x.data<float>(),
      wq.data<uint8_t>(),
      scales.data<float>(),
      biases ? biases->data<float>() : nullptr,
      out.data<float>(),
      M,
      N,
      K,
      group_size,
      bits);
  // TODO: Add error checking, support for other dtypes, and real dequant logic
}

} // namespace mlx::core
