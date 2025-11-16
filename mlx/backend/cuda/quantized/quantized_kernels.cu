// Copyright © 2025 Apple Inc.

#include <cuda_runtime.h>
#include <cstdint>
#include "mlx/backend/cuda/quantized/quantized_kernels.cuh"

namespace mlx::core {

// Placeholder CUDA kernel for qmv
__global__ void quantized_qmv_kernel(
    const float* x,
    const uint8_t* wq,
    const float* scales,
    const float* biases,
    float* out,
    int M,
    int N,
    int K,
    int group_size,
    int bits) {
  // TODO: Implement real qmv logic
}

void quantized_qmv_cuda(
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
  quantized_qmv_kernel<<<grid, block, 0, enc.stream()>>>(
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
}

// Placeholder CUDA kernel for qvm
__global__ void quantized_qvm_kernel(
    const float* x,
    const uint8_t* wq,
    const float* scales,
    const float* biases,
    float* out,
    int M,
    int N,
    int K,
    int group_size,
    int bits) {
  // TODO: Implement real qvm logic
}

void quantized_qvm_cuda(
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
  quantized_qvm_kernel<<<grid, block, 0, enc.stream()>>>(
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
}

// Placeholder CUDA kernel for qmv_quad
__global__ void quantized_qmv_quad_kernel(
    const float* x,
    const uint8_t* wq,
    const float* scales,
    const float* biases,
    float* out,
    int M,
    int N,
    int K,
    int group_size,
    int bits) {
  // TODO: Implement real qmv_quad logic
}

void quantized_qmv_quad_cuda(
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
  quantized_qmv_quad_kernel<<<grid, block, 0, enc.stream()>>>(
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
}

// Placeholder CUDA kernel for qvm_split_k
__global__ void quantized_qvm_split_k_kernel(
    const float* x,
    const uint8_t* wq,
    const float* scales,
    const float* biases,
    float* out,
    int M,
    int N,
    int K,
    int group_size,
    int bits) {
  // TODO: Implement real qvm_split_k logic
}

void quantized_qvm_split_k_cuda(
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
  quantized_qvm_split_k_kernel<<<grid, block, 0, enc.stream()>>>(
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
}

} // namespace mlx::core
