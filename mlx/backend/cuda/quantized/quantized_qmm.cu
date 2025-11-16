// Copyright © 2025 Apple Inc.

#include <cuda_runtime.h>
#include <cstdint>
#include "mlx/backend/cuda/quantized/quantized_qmm.cuh"

namespace mlx::core {





// Helper: get pack factor (number of elements per uint32)
__device__ inline int get_pack_factor(int bits) {
  if (bits == 3 || bits == 5) return 8;
  if (bits == 6) return 4;
  return 32 / bits;
}


// Helper: unpack quantized value from packed buffer (matches CPU logic for all bits)
__device__ inline int unpack_qval(const uint8_t* wq, int group_size, int bits, int k, int col, int N, int K) {
  int group = col / group_size;
  int group_col = col % group_size;
  int pack_factor = get_pack_factor(bits);
  int packs_in_group = group_size / pack_factor;
  int val = 0;
  if (bits == 3) {
    int bytes_per_pack = 3;
    int pack_idx = group_col / pack_factor;
    int p = group_col % pack_factor;
    int base_offset = group * K * packs_in_group * bytes_per_pack + k * packs_in_group * bytes_per_pack;
    const uint8_t* w = wq + base_offset + (pack_idx * 3);
    if (p == 0) val = w[0] & 0x7;
    else if (p == 1) val = (w[0] & 0x38) >> 3;
    else if (p == 2) val = ((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2);
    else if (p == 3) val = (w[1] & 0xe) >> 1;
    else if (p == 4) val = (w[1] & 0x70) >> 4;
    else if (p == 5) val = ((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1);
    else if (p == 6) val = (w[2] & 0x1c) >> 2;
    else if (p == 7) val = (w[2] & 0xe0) >> 5;
  } else if (bits == 5) {
    int bytes_per_pack = 5;
    int pack_idx = group_col / pack_factor;
    int p = group_col % pack_factor;
    int base_offset = group * K * packs_in_group * bytes_per_pack + k * packs_in_group * bytes_per_pack;
    const uint8_t* w = wq + base_offset + (pack_idx * 5);
    if (p == 0) val = w[0] & 0x1f;
    else if (p == 1) val = ((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3);
    else if (p == 2) val = (w[1] & 0x7c) >> 2;
    else if (p == 3) val = ((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1);
    else if (p == 4) val = ((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4);
    else if (p == 5) val = (w[3] & 0x3e) >> 1;
    else if (p == 6) val = ((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2);
    else if (p == 7) val = (w[4] & 0xf8) >> 3;
  } else if (bits == 6) {
    int bytes_per_pack = 4;
    int pack_idx = group_col / pack_factor;
    int p = group_col % pack_factor;
    int base_offset = group * K * packs_in_group * bytes_per_pack + k * packs_in_group * bytes_per_pack;
    const uint8_t* w = wq + base_offset + (pack_idx * 3);
    if (p == 0) val = w[0] & 0x3f;
    else if (p == 1) val = ((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2);
    else if (p == 2) val = ((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4);
    else if (p == 3) val = (w[2] >> 2) & 0x3f;
  }
  if (bits == 2 || bits == 4 || bits == 8) {
    int pack_factor = get_pack_factor(bits);
    int packs_in_group = group_size / pack_factor;
    int group = col / group_size;
    int group_col = col % group_size;
    int pack_idx = group_col / pack_factor;
    int p = group_col % pack_factor;
    // group-major addressing: [group][k][pack]
    int base_offset = group * K * packs_in_group + k * packs_in_group + pack_idx;
    uint8_t wi = wq[base_offset];
    val = (wi >> (p * bits)) & ((1 << bits) - 1);
#ifdef DEBUG_QMM
    if (col < 4 && k < 4) {
      printf("[QMM DEBUG] col=%d k=%d group=%d group_col=%d pack_idx=%d p=%d base_offset=%d wi=0x%02x val=%d\n", col, k, group, group_col, pack_idx, p, base_offset, wi, val);
    }
#endif
  } else {
    val = 0;
  }
  return val;
}

// CUDA kernel for quantized matmul (affine, non-batched, non-transposed)


__global__ void quantized_qmm_kernel(
    const float* x, // [M, K]
  const uint8_t* wq, // packed
    const float* scales, // [N/group_size]
    const float* biases, // [N/group_size]
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
    int group = col / group_size;
    float scale = scales[group];
    float bias = biases ? biases[group] : 0.0f;
    int n_groups = N / group_size;
    int pack_factor = get_pack_factor(bits);
    int packed_size = n_groups * K * group_size / pack_factor;
    for (int k = 0; k < K; ++k) {
  int qval = unpack_qval(wq, group_size, bits, k, col, N, K);
      float w = scale * qval + bias;
      acc += x[row * K + k] * w;
    }
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
  int N = out.shape(-1);
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
  // TODO: Add error checking, support for other dtypes, and optimize memory access
}

} // namespace mlx::core
