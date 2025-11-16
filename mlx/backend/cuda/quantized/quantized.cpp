// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include <nvtx3/nvtx3.hpp>
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/quantized_kernels.cuh"
#include "mlx/backend/cuda/quantized/quantized_qmm.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace {

inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

} // namespace

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Quantize::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(cu::malloc_async(w.nbytes(), enc.stream()));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      fp_dequantize(wq, scales, w, group_size_, bits_, enc, s);
    }
  } else {
    auto w = ensure_row_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc.stream()));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc.stream()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc.stream()));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      fp_quantize(w, wq, scales, group_size_, bits_, enc, s);
    }
  }
}

// CUDA implementation of QuantizedMatmul::eval_gpu
void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(cu::malloc_async(out.nbytes(), enc.stream()));

  // Ensure inputs are row-contiguous for efficient CUDA access
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array wq = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  // Extract the matmul shapes
  bool non_batched = wq.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  // Routing logic matching Metal backend
  int vector_limit = transpose_ ? 4 : 4; // TODO: tune for CUDA
  // It is a matrix matrix product.
  if (M >= vector_limit) {
    quantized_qmm_cuda(
        x,
        wq,
        scales,
        biases ? &(*biases) : nullptr,
        out,
        group_size_,
        bits_,
        enc,
        s);
    return;
  }

  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if (transpose_ && (K == 128 || K == 64)) {
    quantized_qmv_quad_cuda(
        x,
        wq,
        scales,
        biases ? &(*biases) : nullptr,
        out,
        group_size_,
        bits_,
        enc,
        s);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    quantized_qmv_cuda(
        x,
        wq,
        scales,
        biases ? &(*biases) : nullptr,
        out,
        group_size_,
        bits_,
        enc,
        s);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    quantized_qvm_cuda(
        x,
        wq,
        scales,
        biases ? &(*biases) : nullptr,
        out,
        group_size_,
        bits_,
        enc,
        s);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  quantized_qvm_split_k_cuda(
      x,
      wq,
      scales,
      biases ? &(*biases) : nullptr,
      out,
      group_size_,
      bits_,
      enc,
      s);
  return;
}

} // namespace mlx::core
