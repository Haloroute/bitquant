from torch import Tensor
from .base import WeightQuantizer


class BinaryWeightQuantizer(WeightQuantizer):
    """Quantizes weights to binary values {-1, 1}.

    Implements the Straight-Through Estimator (STE) in the backward pass to 
    allow gradient flow through the non-differentiable rounding operation.
    """

    def quantize(self, weight: Tensor) -> Tensor:
        """Executes the core quantization mapping for weights.

        Args:
            weight (Tensor): The continuous-valued weight tensor.

        Returns:
            Tensor: The binary-quantized weight tensor, scaled by the mean absolute value.
        """
        gamma = weight.abs().mean()
        alpha = weight.mean()
        weight_q = (weight > alpha).float() * 2 - 1
        return weight_q * gamma


class TernaryWeightQuantizer(WeightQuantizer):
    """Quantizes weights to ternary values {-1, 0, 1}.

    Implements the Straight-Through Estimator (STE) in the backward pass to 
    allow gradient flow through the non-differentiable rounding operation.
    """

    def quantize(self, weight: Tensor) -> Tensor:
        """Executes the core quantization mapping for weights.

        Args:
            weight (Tensor): The continuous-valued weight tensor.

        Returns:
            Tensor: The ternary-quantized weight tensor, scaled by the mean absolute value.
        """
        gamma = weight.abs().mean().clamp(min=1e-8)
        weight_scaled = weight / gamma
        weight_q = weight_scaled.round().clamp(-1, 1)
        return weight_q * gamma