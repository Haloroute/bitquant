<div align="center">

# ⚡ BitQuant

**Zero-Intrusion, Multi-Precision Quantization Wrappers for PyTorch**

<!-- [![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

</div>

---

`bitquant` allows you to apply advanced quantization techniques (like BitNet b1.58, INT8, INT4, and Binary weights) to your existing PyTorch models **without altering the internal layer implementations**. Instead of replacing your `nn.Linear` or `nn.Conv2d` layers with custom quantized variants, `bitquant` dynamically wraps your standard layers.

## ✨ Key Features

- **Non-Invasive Architecture:** Wraps around standard PyTorch modules using native `torch.nn.utils.parametrize` for weights and `forward_pre_hook` for activations.
- **Straight-Through Estimator (STE):** Fully supports training! Non-differentiable rounding operations are bypassed in the backward pass using STE, allowing gradients to flow to the original weights.
- **Clean State Checkpoints:** Custom `state_dict` hooks ensure that your quantized model saves and loads weights seamlessly without breaking your model's original parameter keys.
- **Modular & Extensible:** Mix and match different weight and activation quantizers, or easily write your own by subclassing the base classes.
- **BETA: Kernel For W1.58A8 GEMV Operation:** Native kernel using `AVX2` for 1.58-bit (I2_S) weight / 8-bit activation GEMV. Still in development.

## 📦 Available Quantizers

| Quantizer Class           | Type       | Description                                                |
| :------------------------ | :--------- | :--------------------------------------------------------- |
| `TernaryWeightQuantizer`  | Weight     | Quantizes weights to ternary {-1, 0, 1}.                   |
| `BinaryWeightQuantizer`   | Weight     | Quantizes weights to binary {-1, 1}.                       |
| `Int8ActivationQuantizer` | Activation | Quantizes input activations to 8-bit integers [-128, 127]. |
| `Int4ActivationQuantizer` | Activation | Quantizes input activations to 4-bit integers [-8, 7].     |

## ⚙️ Available Kernels

| Kernel Name         | Target ISA | Format           | Description                                                                |
| :------------------ | :--------- | :--------------- | :------------------------------------------------------------------------- |
| `gemm_avx2_w1.58a8` | AVX2       | W1.58A8 (`I2_S`) | Vectorized GEMM kernel for ternary weights and 8-bit activations via AVX2. |

## 📦 Installation

Install from source in editable mode:

```bash
git clone https://github.com/haloroute/bitquant.git
cd bitquant
pip install -e .
```

## 🚀 Quick Start

Here is an example of how to convert a standard Convolutional Neural Network into a **BitNet b1.58** model in just a few lines of code.

```python
import torch
import torch.nn as nn
from bitquant import (
    Quantizer,
    TernaryWeightQuantizer,
    Int8ActivationQuantizer
)

# 1. Define your standard PyTorch model normally
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = torch.flatten(x, 1)
        return self.fc1(x)

model = SimpleCNN()

# 2. Initialize the Quantizer
# (e.g., BitNet b1.58 uses Ternary weights and Int8 Activations)
quantizer = Quantizer(
    weight_quantizer=TernaryWeightQuantizer,
    activation_quantizer=Int8ActivationQuantizer
)

# 3. Apply quantization wrapper directly to the layers!
quantizer.wrap(model.conv1)
quantizer.wrap(model.fc1)

# Your model is now a quantized BitNet b1.58 model!
# It is ready for a forward pass or training loop.
dummy_inputs = torch.randn(4, 1, 28, 28)
output = model(dummy_inputs)
```

And here is an example of using custom kernel to calculate GEMV between 1.58-bit (`I2_S`) matrix and 8-bit vector:

```python
import numpy as np
from bitquant.kernel import pack_matrix, compute_gemv

# 1. Define matrix dimensions
n_rows, n_cols = 128, 128

# 2. Initialize input data
# Weights: ternary values {-1, 0, 1}
weights_np = np.random.choice([-1, 0, 1], size=(n_rows, n_cols)).astype(np.int8)
# Activations: 8-bit integers [-127, 127]
activation_np = np.random.randint(-127, 127, size=n_cols, dtype=np.int8)

# 3. Pack the weights for the W1.58A8 format
weights_packed = pack_matrix(weights_np)

# 4. Execute the AVX2 GEMV kernel
result = compute_gemv(weights_packed, activation_np, n_rows, n_cols)

print("Result shape:", result.shape)
```

## 🛠️ How it Works Under the Hood

Instead of subclassing PyTorch layers and breaking parameter naming conventions, `bitquant` uses an outer wrapper pattern:

```
                  ┌───────────────────────────────┐
                  │           Quantizer           │
 Input Tensor ───►│                               │───► Output Tensor
                  │   ┌───────────────────────┐   │
                  │   │    Original Layer     │   │
                  │   │   (e.g., nn.Linear)   │   │
                  │   └───────────────────────┘   │
                  │               ▲               │
                  │       Quantization Hook       │
                  └───────────────────────────────┘
```

When you call `quantizer.wrap(module)`:

1. **Weights:** It registers a parameterization via `torch.nn.utils.parametrize`. PyTorch will automatically pass the continuous weight tensor through the `WeightQuantizer` (e.g., scaling and rounding to -1, 0, 1) right before the layer performs its forward computation.
2. **Activations:** It registers a `forward_pre_hook` to the module. When an input tensor arrives at the layer, it is intercepted, quantized by the `ActivationQuantizer`, and passed into the layer.
3. **State Management:** Custom hooks are registered so that if you call `model.state_dict()`, the original unquantized float weights are safely mapped back to their original dictionary keys, making checkpointing painless.

When you call `quantizer.unwrap(module, leave_quantized=<leave_quantized>)`:

1. **Weights:** It safely removes the parameterization wrapper via `torch.nn.utils.parametrize.remove_parametrizations`.
   - If `leave_quantized=True`, the quantized weights (e.g., -1, 0, 1) are permanently "baked" into the parameter, which is ideal for final inference and deployment.
   - If `leave_quantized=False`, the layer seamlessly reverts to using its original continuous, high-precision weights.
2. **Activations & State Management:** It iterates through saved hook handles to cleanly deregister the activation `forward_pre_hook` and all custom `state_dict` hooks. The module is entirely restored to its native PyTorch state without leaving any residual operations behind.

## 🗺️ Roadmap

- [x] Non-invasive layer wrapping architecture using `torch.nn.utils.parametrize` and forward pre-hooks.
- [x] Ternary and binary weight quantization with Straight-Through Estimator (STE) support.
- [x] INT8 and INT4 activation quantization with Straight-Through Estimator (STE) support.
- [x] `unwrap` function to restore the converted quantized model back to its original.
- [ ] INT8 and INT4 weight quantization with Straight-Through Estimator (STE) support.
- [ ] Custom kernels for BitNet b1.58 Linear/Conv operations, compatible with `ONNX`/`PyTorch` on CPU.
  - [x] AVX2 kernel for GEMV operation (W1.58A8), compatible with `numpy` array.
  - [ ] AVX2 kernel for GEMM/Conv operation (W1.58A8).
- [ ] Custom kernels for BitNet b1.58 Linear/Conv operations, compatible with `ONNX`/`PyTorch` on NVIDIA GPU.

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! If you want to add support for new quantization formats (like FP8 or NF4), simply create a new class extending `WeightQuantizer` or `ActivationQuantizer` in `weight.py` or `activation.py`.
