#include "gemv.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Wrapper function for the AVX2 GEMV operation
nb::ndarray<int32_t, nb::numpy, nb::c_contig> compute_gemv(
    nb::ndarray<uint8_t, nb::numpy, nb::ndim<2>, nb::c_contig> weight,
    nb::ndarray<int8_t, nb::numpy, nb::ndim<1>, nb::c_contig> activation,
    int n_rows, int n_cols) {
  // 1. Allocate raw memory in C++ for the output array
  int32_t *data = new int32_t[n_rows];

  // 2. Execute the AVX2 C++ kernel (writes directly to the allocated data
  // array)
  gemv_i2_s((const uint8_t *)weight.data(), (const int8_t *)activation.data(),
            data, n_rows, n_cols);

  // 3. Define the shape of the output array (1D array with size n_rows)
  size_t shape[1] = {static_cast<size_t>(n_rows)};

  // 4. Create a capsule to manage the memory lifecycle
  // This tells Python's garbage collector how to safely delete the C++ array
  nb::capsule owner(data, [](void *p) noexcept { delete[] (int32_t *)p; });

  // 5. Initialize and return a zero-copy NumPy array
  return nb::ndarray<int32_t, nb::numpy, nb::c_contig>(
      data,  // Data pointer
      1,     // Number of dimensions
      shape, // Shape array
      owner  // Memory manager capsule
  );
}

// Define the Python module for GEMV and expose the functions
NB_MODULE(_gemv, m) {
  m.doc() = "AVX2 optimized GEMV operations and packing utilities via nanobind";

  m.def("compute_gemv", &compute_gemv,
        "Compute GEMV with I2_S weights and INT8 activations",
        nb::arg("weight"), nb::arg("activation"), nb::arg("n_rows"),
        nb::arg("n_cols"));
}