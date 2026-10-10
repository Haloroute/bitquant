from torch import Tensor
from .base import ActivationQuantizer


class Int8ActivationQuantizer(ActivationQuantizer):
    """Quantizes activations to 8-bit integers [-128, 127] (BitNet b1.58).

    Uses per-token absmax scaling: each token (row of the last dimension) is scaled
    by 127 / max(|x|), so the quantized values effectively lie in [-127, 127].

    Implements the Straight-Through Estimator (STE) in the backward pass.
    """

    def quantize(cls, x: Tensor, dim: int = None) -> Tensor:
        """Executes the core quantization mapping for activations.

        Args:
            x (Tensor): The input activation tensor.
            dim (int, optional): The dimension along which to compute the quantization scales. If None, global absmax scaling is used. Defaults to None.

        Returns:
            Tensor: The 8-bit quantized activation tensor, scaled to preserve magnitude.
        """
        if dim is None:
            # Global absmax scaling
            gamma_x = x.abs().max().clamp(min=1e-5)
        else:
            # Per token absmax scaling
            gamma_x = x.abs().max(dim=dim, keepdim=True).values.clamp(min=1e-5)
 
        scale = 127.0 / gamma_x
        x_scaled = x * scale
        x_q = x_scaled.round().clamp(-128, 127)
        return x_q / scale


class Int4ActivationQuantizer(ActivationQuantizer):
    """Quantizes activations to 4-bit integers [-8, 7] (BitNet b1.58 style).

    Uses per-token absmax scaling: each token (row of the last dimension) is scaled
    by 7 / max(|x|), so the quantized values effectively lie in [-7, 7].

    Implements the Straight-Through Estimator (STE) in the backward pass.
    """

    def quantize(cls, x: Tensor, dim: int = None) -> Tensor:
        """Executes the core quantization mapping for activations.

        Args:
            x (Tensor): The input activation tensor.
            dim (int, optional): The dimension along which to compute the quantization scales. If None, global absmax scaling is used. Defaults to None.

        Returns:
            Tensor: The 4-bit quantized activation tensor, scaled to preserve magnitude.
        """
        if dim is None:
            # Global absmax scaling
            gamma_x = x.abs().max().clamp(min=1e-5)
        else:
            # Per token absmax scaling
            gamma_x = x.abs().max(dim=dim, keepdim=True).values.clamp(min=1e-5)

        scale = 7.0 / gamma_x
        x_scaled = x * scale
        x_q = x_scaled.round().clamp(-8, 7)
        return x_q / scale