from torch import Tensor
from .base import ActivationQuantizer


class Int8ActivationQuantizer(ActivationQuantizer):
    """Quantizes activations to 8-bit integers [-128, 127].

    Implements the Straight-Through Estimator (STE) in the backward pass.
    """

    def quantize(cls, x: Tensor) -> Tensor:
        """Executes the core quantization mapping for activations.

        Args:
            x (Tensor): The input activation tensor.

        Returns:
            Tensor: The 8-bit quantized activation tensor, scaled to preserve magnitude.
        """
        gamma_x = x.abs().max().clamp(min=1e-8)
        scale = 127.0 / gamma_x
        x_scaled = x * scale
        x_q = x_scaled.round().clamp(-128, 127)
        return x_q / scale


class Int4ActivationQuantizer(ActivationQuantizer):
    """Quantizes activations to 4-bit integers [-8, 7].

    Implements the Straight-Through Estimator (STE) in the backward pass.
    """

    def quantize(cls, x: Tensor) -> Tensor:
        """Executes the core quantization mapping for activations.

        Args:
            x (Tensor): The input activation tensor.

        Returns:
            Tensor: The 4-bit quantized activation tensor, scaled to preserve magnitude.
        """
        gamma_x = x.abs().max().clamp(min=1e-8)
        scale = 7.0 / gamma_x
        x_scaled = x * scale
        x_q = x_scaled.round().clamp(-8, 7)
        return x_q / scale