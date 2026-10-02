#include "pack.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Wrapper function to pack int8 W1.58 weights into I2_S format
nb::ndarray<uint8_t, nb::numpy, nb::c_contig>
pack_matrix(nb::ndarray<int8_t, nb::numpy, nb::ndim<2>, nb::c_contig> input) {
  // 1. Extract dimensions from the input array
  int n_rows = input.shape(0);
  int n_cols = input.shape(1);

  // 2. Calculate the output dimensions based on the I2_S packing rules
  int n_cols_padded = (n_cols + 127) / 128 * 128;
  int out_cols = n_cols_padded / 4; // 4 elements are packed into 1 byte

  // 3. Allocate raw memory for the output array
  uint8_t *out_data = new uint8_t[n_rows * out_cols];

  // 4. Call the core C++ packing function
  pack_2d_i2s((const int8_t *)input.data(), out_data, n_rows, n_cols);

  // 5. Create a capsule to manage the memory lifecycle
  nb::capsule owner(out_data, [](void *p) noexcept { delete[] (uint8_t *)p; });

  // 6. Initialize and return a zero-copy 2D NumPy array
  size_t shape[2] = {static_cast<size_t>(n_rows),
                     static_cast<size_t>(out_cols)};
  return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(out_data, 2, shape,
                                                       owner);
}

// Define the Python module for packing and expose the function
NB_MODULE(_pack, m) {
  m.doc() = "Packing utilities for int8 weight matrices into I2_S format";

  m.def("pack_matrix", &pack_matrix,
        "Pack an int8 weight matrix (W1.58) into I2_S format",
        nb::arg("input"));
}