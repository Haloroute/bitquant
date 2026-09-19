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
        # Store the quantizer classes for later use
        self.weight_quantizer_class = weight_quantizer
        self.activation_quantizer_class = activation_quantizer

        # Dictionary to store hook handles for each module
        self._hook_handles = {}

    def wrap(self, module: nn.Module, name_: str = 'weight') -> nn.Module:
        """Applies quantization in-place to a PyTorch layer.

        Attaches a weight parametrization for ternary weights and a forward 
        pre-hook for 8-bit input activations.

        Args:
            module (nn.Module): The PyTorch layer to be quantized (e.g., nn.Linear). This layer must have a weight parameter to be quantized and only one input tensor to be quantized.
            name_ (str): The name of the weight parameter in the module (default is 'weight').

        Returns:
            nn.Module: The modified module with BitNet b1.58 operations attached.
        """
        # Store handles for hooks to allow later removal
        handles = []

        if hasattr(module, name_) and isinstance(getattr(module, name_), Tensor):
            if not parametrize.is_parametrized(module, name_):
                # 1. Register the parametrization
                weight_quantizer = self.weight_quantizer_class()
                parametrize.register_parametrization(module, name_, weight_quantizer)

                # 2. Hook to rename the key back to normal when saving the state_dict
                def state_dict_hook(mod, state_dict, prefix, local_metadata):
                    param_key = f"{prefix}parametrizations.{name_}.original"
                    orig_key = f"{prefix}{name_}"
                    if param_key in state_dict:
                        # Move the unquantized weight back to the original key
                        state_dict[orig_key] = state_dict.pop(param_key)
                    return state_dict

                # 3. Hook to convert the key back to the parametrized version when loading
                def load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                    param_key = f"{prefix}parametrizations.{name_}.original"
                    orig_key = f"{prefix}{name_}"
                    if orig_key in state_dict:
                        # Map the incoming original key to the parametrized location
                        state_dict[param_key] = state_dict.pop(orig_key)

                # 4. Attach the hooks to the module
                handles.append(module._register_state_dict_hook(state_dict_hook))

                if hasattr(module, 'register_load_state_dict_pre_hook'):
                    handles.append(module.register_load_state_dict_pre_hook(load_state_dict_pre_hook))
                else:
                    handles.append(module._register_load_state_dict_pre_hook(load_state_dict_pre_hook))

        # Only apply activation quantization to leaf modules
        if len(list(module.children())) == 0:
            activation_quantizer = self.activation_quantizer_class()
            module.register_forward_pre_hook(activation_quantizer.pre_hook)

        # Store all handles associated with this module
        if handles:
            self._hook_handles[module] = handles

        return module

    def unwrap(self, module: nn.Module, name_: str = 'weight', leave_quantized: bool = False) -> nn.Module:
        """Removes quantization parametrizations and hooks from a module.

        Args:
            module: The quantized module to unwrap.
            name_: The name of the parametrized tensor (default 'weight').
            leave_quantized: If False, restores the original continuous weights. 
                             If True, bakes in the quantized weights permanently.
        """
        # 1. Remove the weight parametrization
        if parametrize.is_parametrized(module, name_):
            parametrize.remove_parametrizations(module, name_, leave_parametrized=leave_quantized)

        # 2. Remove all registered hooks (state_dict and forward pre-hooks)
        if module in self._hook_handles:
            for handle in self._hook_handles.pop(module):
                handle.remove()

        return module