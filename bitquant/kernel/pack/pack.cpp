#include "pack.h"
#include <cstring>

// ============================================================
// Function to pack a 1.58-bit weight matrix in INT8 format into I2_S format
// Conversion: ternary int8 weights {-1,0,1} -> GGML I2_S (2 bits/weight, stored
// as w+1). Per 128-column block: byte i holds cols i, i+32, i+64, i+96 at bits
// [7:6],[5:4],[3:2],[1:0]. Output size: n_rows * ceil(n_cols / 128) * 32 bytes.
// ============================================================
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

// ============================================================
// Function to pack a INT4 activation vector in INT8 format into A4 format
// Conversion: INT4 activations (int8 values in [-7, 7]) -> packed INT4 for
// gemv_i2_s_a4. Every 32 bytes hold 64 values: byte i = A[i] << 4 | A[i + 32]
// (two's-complement nibbles). Nibbles past n_cols are 0. The output is
// zero-padded to a multiple of 128 columns (64 bytes) so it lines up with the
// I2_S weight blocks. Output size: ceil(n_cols / 128) * 64 bytes.
// ============================================================
void pack_1d_int4(const int8_t* act_in, uint8_t* act_out, int n_cols) {
    const int n_cols_padded = (n_cols + 127) & ~127;
    auto nib = [&](int col) -> uint8_t {
        return (col < n_cols) ? (static_cast<uint8_t>(act_in[col]) & 0x0F) : 0;
        };
    for (int cb = 0; cb < n_cols_padded; cb += 64) {
        uint8_t* out = act_out + cb / 2;
        for (int i = 0; i < 32; ++i)
            out[i] = static_cast<uint8_t>((nib(cb + i) << 4) | nib(cb + i + 32));
    }
}