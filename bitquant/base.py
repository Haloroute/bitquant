import torch.nn as nn

from torch import Tensor
from torch.nn.utils import parametrize
from typing import Type


class WeightQuantizer(nn.Module):
    """This class serves as a template for implementing specific quantization strategies for weights.
    The 'quantize' method should be overridden in subclasses to define the quantization mapping.

    Implements the Straight-Through Estimator (STE) to allow gradient flow through the non-differentiable rounding operation.
    """

    def quantize(self, weight: Tensor) -> Tensor:
        """Executes the core quantization mapping for weights.

        Args:
            weight (Tensor): The continuous-valued weight tensor.

        Returns:
            Tensor: The quantized weight tensor.
        """
        return weight

    def forward(self, weight: Tensor) -> Tensor:
        """Applies quantization with the Straight-Through Estimator (STE).

        Args:
            weight (Tensor): The continuous-valued weight tensor.

        Returns:
            Tensor: The quantized weight tensor with STE gradient flow.
        """
        weight_q = self.quantize(weight)
        return weight + (weight_q - weight).detach()


class ActivationQuantizer(nn.Module):
    """This class serves as a template for implementing specific quantization strategies for activations.
    The 'quantize' method should be overridden in subclasses to define the quantization mapping.

    Implements the Straight-Through Estimator (STE) to allow gradient flow through the non-differentiable rounding operation.
    """

    def quantize(self, x: Tensor) -> Tensor:
        """Executes the core quantization mapping for activations.

        Args:
            x (Tensor): The input activation tensor.

        Returns:
            Tensor: The quantized activation tensor.
        """
        return x

    def apply_ste(self, x: Tensor) -> Tensor:
        """Applies quantization with the Straight-Through Estimator (STE).

        Args:
            x (Tensor): The input activation tensor.

        Returns:
            Tensor: The quantized activation tensor with STE gradient flow.
        """
        x_q = self.quantize(x)
        return x + (x_q - x).detach()

    def forward(self, x: Tensor) -> Tensor:
        """Executes activation quantization on the input tensor.

        Args:
            x (Tensor): The input activation tensor.

        Returns:
            Tensor: The quantized activation tensor.
        """
        return self.apply_ste(x)

    def pre_hook(self, module: nn.Module, args: tuple) -> tuple:
        """A forward pre-hook that quantizes layer inputs.

        Args:
            module (nn.Module): The layer module being executed.
            args (tuple): Input arguments passed to the module.

        Returns:
            tuple: Transformed arguments containing quantized activations.
        """
        if not args:
            return args
        return (self.apply_ste(args[0]), *args[1:])


class Quantizer():
    """Base class for applying quantization to layers.

    This class serves as a template for implementing specific quantization strategies for different types of layers.
    """

    def __init__(self, weight_quantizer: Type[WeightQuantizer], activation_quantizer: Type[ActivationQuantizer]):
        """Initializes the Quantizer with given quantizer classes.

        Args:
            weight_quantizer (Type[WeightQuantizer]): The weight quantizer class to be applied to layers.
            activation_quantizer (Type[ActivationQuantizer]): The activation quantizer class to be applied to layers.
        """
        self.weight_quantizer_class = weight_quantizer
        self.activation_quantizer_class = activation_quantizer

    def apply(self, module: nn.Module, name_: str = 'weight') -> nn.Module:
        """Applies quantization in-place to a PyTorch layer.

        Attaches a weight parametrization for ternary weights and a forward 
        pre-hook for 8-bit input activations.

        Args:
            module (nn.Module): The PyTorch layer to be quantized (e.g., nn.Linear). This layer must have a weight parameter to be quantized and only one input tensor to be quantized.
            name_ (str): The name of the weight parameter in the module (default is 'weight').

        Returns:
            nn.Module: The modified module with BitNet b1.58 operations attached.
        """
        if hasattr(module, name_) and isinstance(getattr(module, name_), Tensor):
            if not parametrize.is_parametrized(module, name_):
                weight_quantizer = self.weight_quantizer_class()
                parametrize.register_parametrization(module, name_, weight_quantizer)

        # Only apply activation quantization to leaf modules to prevent double-quantization
        if len(list(module.children())) == 0:
            activation_quantizer = self.activation_quantizer_class()
            module.register_forward_pre_hook(activation_quantizer.pre_hook)
            
        return module