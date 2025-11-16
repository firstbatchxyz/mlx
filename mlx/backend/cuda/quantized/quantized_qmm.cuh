// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

// Launch a simple quantized matmul CUDA kernel (affine, non-batched,
// non-transposed)
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

} // namespace mlx::core
