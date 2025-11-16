// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

// CUDA kernel launcher for quantized matrix-matrix multiply (qmm)
void quantized_qmm_cuda(
    const array& x,
    const array& wq,
    const array& scales,
    const array* biases,
    array& out,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

// CUDA kernel launcher for quantized matrix-vector multiply (qmv)
void quantized_qmv_cuda(
    const array& x,
    const array& wq,
    const array& scales,
    const array* biases,
    array& out,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

// CUDA kernel launcher for quantized vector-matrix multiply (qvm)
void quantized_qvm_cuda(
    const array& x,
    const array& wq,
    const array& scales,
    const array* biases,
    array& out,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

// CUDA kernel launcher for quantized matrix-vector quad multiply (qmv_quad)
void quantized_qmv_quad_cuda(
    const array& x,
    const array& wq,
    const array& scales,
    const array* biases,
    array& out,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

// CUDA kernel launcher for quantized vector-matrix split-k multiply
// (qvm_split_k)
void quantized_qvm_split_k_cuda(
    const array& x,
    const array& wq,
    const array& scales,
    const array* biases,
    array& out,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
