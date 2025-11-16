import mlx.core as mx
from typing import Optional
import pytest

# t = mx.load("mlx_quantization")
#mx.set_default_device(mx.cpu)

def quantized_matmul_new(
    x: mx.array,
    w: mx.array,
    /,
    scales: mx.array,
    biases: Optional[mx.array] = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    stream=None,  # JAX doesn't use streams like MLX
) -> mx.array:
    """
    Perform matrix multiplication with quantized matrix w.

    The quantization uses one floating point scale and bias per group_size elements.
    Each element in w takes `bits` bits and is packed in an unsigned 32 bit integer.

    Args:
        x: Input array of shape (..., in_features)
        w: Quantized matrix packed in uint32, shape (in_features // group_size * group_size // elements_per_int, out_features)
        scales: Scales per group, shape (out_features, in_features // group_size)
        biases: Optional biases per group, same shape as scales
        transpose: If True, compute x @ w.T, else x @ w
        group_size: Number of elements sharing one scale/bias
        bits: Bits per quantized element (typically 4 or 8)
        mode: Quantization mode ('affine' for scale+bias, 'symmetric' for scale only)

    Returns:
        Result of matrix multiplication with shape (..., out_features)
    """
    if mode not in ["affine", "symmetric"]:
        raise ValueError(f"mode must be 'affine' or 'symmetric', got {mode}")

    # Dequantize weights using MLX's dequantization function
    w_dequant = mx.dequantize(
        w, scales, biases, bits=bits, group_size=group_size, mode=mode, stream=stream
    )

    x_dequant = x.astype(dtype=w_dequant.dtype, stream=stream)
    if transpose:
        result = mx.matmul(x_dequant, mx.transpose(w_dequant))
    else:
        result = mx.matmul(x_dequant, w_dequant)

    result_quant = result.astype(dtype=x.dtype, stream=stream)
    return result_quant


def get_quantized_wgt(
    w_shape: tuple[int, ...],
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    stream=None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Quantize the weight matrix w.

    Args:
        w: Weight matrix of shape (in_features, out_features)
        group_size: Number of elements sharing one scale/bias
        bits: Bits per quantized element (typically 4 or 8)
        mode: Quantization mode ('affine' for scale+bias, 'symmetric' for scale only)

    Returns:
        A tuple of (packed_weights, scales, biases)
    """
    if mode not in ["affine", "symmetric"]:
        raise ValueError(f"mode must be 'affine' or 'symmetric', got {mode}")

    if bits not in [2, 4, 8]:
        raise ValueError(f"bits must be 2, 4, or 8, got {bits}")

    # Create random weights
    w = mx.random.uniform(shape=w_shape, dtype=mx.float32, low=-1.0, high=1.0)

    # Quantize weights using MLX's quantization function
    packed_w, scales, biases = mx.quantize(
        w, bits=bits, group_size=group_size, mode=mode, stream=stream
    )

    return packed_w, scales, biases


# @pytest.mark.parametrize("bits", [2, 4, 8])
# @pytest.mark.parametrize("num_batches", [2, 4, 8])
# @pytest.mark.parametrize("out_features", [2, 4, 8, 16, 32, 64])
# @pytest.mark.parametrize("group_size", [32, 64])
# @pytest.mark.parametrize("mul_factor", [2, 4, 8])

@pytest.mark.parametrize("bits", [2,4,8])
@pytest.mark.parametrize("num_batches", [1, 2, 4])
@pytest.mark.parametrize("out_features", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("mul_factor", [2, 4, 8])
def test_simple_matmul(
    bits: int, num_batches: int, out_features: int, group_size: int, mul_factor: int
):
    in_features = group_size * mul_factor

    # prepare weights
    w_shape = (out_features, in_features)
    w_packed, scales, biases = get_quantized_wgt(
        w_shape, bits=bits, group_size=group_size
    )

    # prepare input
    x = mx.random.uniform(
        shape=(num_batches, in_features), dtype=mx.float32, low=-1.0, high=1.0
    )

    print("Input:", x.shape)
    print("Weights shape:", w_shape)
    print("Weights_packed shape:", w_packed.shape)
    print("Scales_packed shape:", scales.shape)
    print("Biases_packed shape:", biases.shape)

    res = mx.quantized_matmul(
        x, w_packed, scales, biases, bits=bits, group_size=group_size
    )
    res_new = quantized_matmul_new(
        x, w_packed, scales, biases, bits=bits, group_size=group_size
    )
    assert res.shape == (num_batches, out_features)
    assert mx.mean(res - res_new) < 1e-5

    # print("Packed weights:", w_packed)
    # print("Scales:", scales)
    # print("Biases:", biases)
    # print("Result:", res)
    print("Output shape:", res.shape)
