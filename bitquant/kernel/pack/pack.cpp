#include "pack.h"
#include <cstring>

// Function to pack a 1.58-bit weight matrix in INT8 format into I2_S format
void pack_2d_i2s(const int8_t *weight, uint8_t *output, int n_rows,
                 int n_cols) {
  // Calculate the number of columns padded to a multiple of 128
  int n_cols_padded = (n_cols + 127) / 128 * 128;

  // Each packed byte contains 4 values, so the output columns are divided by 4
  int out_cols = n_cols_padded / 4;

  for (int r = 0; r < n_rows; ++r) {
    // Iterate through each 128-element block in the current row
    for (int b = 0; b < n_cols_padded / 128; ++b) {
      // Calculate base offsets for the current block
      int in_offset = r * n_cols + b * 128;
      int out_offset = r * out_cols + b * 32;

      // Process 32 elements per sub-block to form 32 packed bytes
      for (int i = 0; i < 32; ++i) {
        // Fetch 4 values with bounds checking. Pad with 0 if out of bounds.
        int8_t v0 = (b * 128 + i < n_cols) ? weight[in_offset + i] : 0;
        int8_t v1 =
            (b * 128 + 32 + i < n_cols) ? weight[in_offset + 32 + i] : 0;
        int8_t v2 =
            (b * 128 + 64 + i < n_cols) ? weight[in_offset + 64 + i] : 0;
        int8_t v3 =
            (b * 128 + 96 + i < n_cols) ? weight[in_offset + 96 + i] : 0;

        // Apply zero-point offset to map W1.58 values {-1, 0, 1} to {0, 1, 2}
        uint8_t m0 = v0 + 1;
        uint8_t m1 = v1 + 1;
        uint8_t m2 = v2 + 1;
        uint8_t m3 = v3 + 1;

        // Shift and interleave the mapped values into a single byte
        // m0 -> bits 6-7, m1 -> bits 4-5, m2 -> bits 2-3, m3 -> bits 0-1
        output[out_offset + i] = (m0 << 6) | (m1 << 4) | (m2 << 2) | m3;
      }
    }
  }
}