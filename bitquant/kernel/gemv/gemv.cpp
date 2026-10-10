#include "gemv.h"
#include "immintrin.h"
#include <cstdint>
#include <cstring>

// =====================================================================================
// Data formats
//   Weight (I2_S)       : 2 bits per value, stored as 0/1/2 meaning -1/0/+1.
//                         32 bytes = 128 values:
//                           bits 7-6 -> cols  0..31, bits 5-4 -> cols  32..63,
//                           bits 3-2 -> cols 64..95, bits 1-0 -> cols  96..127
//   Activation (INT8)   : 1 value per byte, range [-127, 127].
//   Activation (INT4)   : 2 values per byte, range [-7, 7]. Every 32 bytes hold
//   64 values:
//                           bits 7-4 -> cols 0..31, bits 3-0 -> cols 32..63
//                           (byte i = A[i] << 4 | A[i + 32])
//                         Nibbles past n_cols must be 0.
//
//   Math: W_signed = W_u - 1, so dot(W_signed, A) = dot(W_u, A) - sum(A)
// =====================================================================================

// Horizontally sum 8 int32 values in a __m256i vector
static inline int32_t hsum_i32_8(const __m256i a // Vector of 8 int32 values
) {
  const __m128i sum128 =
      _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
  const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
  const __m128i sum64 = _mm_add_epi32(hi64, sum128);
  const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Horizontally sum 4 int64 values in a __m256i vector
static inline int64_t hsum_i64_4(const __m256i a // Vector of 4 int64 values
) {
  const __m128i sum128 =
      _mm_add_epi64(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
  const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
  return _mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64));
}

// -------------------------------------------------------------------------------------
// INT8 helpers
// -------------------------------------------------------------------------------------

// Compute the offset sum of one 128-column block of INT8 activations.
// Each value s is mapped to s + 128 (in [1, 255]) and summed; returns 4 int64
// partial sums.
static inline __m256i sum_block_i8(
    const int8_t
        *py, // Pointer to the block of the activation vector (INT8), 128 bytes
    const __m256i sign_flip // Constant vector to map int8 to uint8 with offset
                            // 128 (0x80 per byte)
) {
  const __m256i zero = _mm256_setzero_si256();

  const __m256i a0 =
      _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)py), sign_flip);
  const __m256i a1 = _mm256_xor_si256(
      _mm256_loadu_si256((const __m256i *)(py + 32)), sign_flip);
  const __m256i a2 = _mm256_xor_si256(
      _mm256_loadu_si256((const __m256i *)(py + 64)), sign_flip);
  const __m256i a3 = _mm256_xor_si256(
      _mm256_loadu_si256((const __m256i *)(py + 96)), sign_flip);

  const __m256i s01 =
      _mm256_add_epi64(_mm256_sad_epu8(a0, zero), _mm256_sad_epu8(a1, zero));
  const __m256i s23 =
      _mm256_add_epi64(_mm256_sad_epu8(a2, zero), _mm256_sad_epu8(a3, zero));
  return _mm256_add_epi64(s01, s23);
}

// Compute the dot product of one 128-column block between I2_S weights and INT8
// activations. Returns 16 int16 lanes; each lane is at most |4 * (2 * 127 + 2 *
// 127)| = 2032.
static inline __m256i dot_block_i2_i8(
    const uint8_t
        *px, // Pointer to the block of the weight matrix (I2_S), 32 bytes
    const int8_t
        *py, // Pointer to the block of the activation vector (INT8), 128 bytes
    const __m256i value_mask // Constant vector to mask out all bits except the
                             // first 2 (0x03 per byte)
) {
  // Unpack weights: 4 bit-planes -> unsigned 0/1/2
  __m256i w3 = _mm256_loadu_si256((const __m256i *)px);
  __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(w3, 2), value_mask);
  __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(w3, 4), value_mask);
  __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(w3, 6), value_mask);
  w3 = _mm256_and_si256(w3, value_mask);

  // Load activations: 1 value per byte
  const __m256i a0 = _mm256_loadu_si256((const __m256i *)py); // cols   0..31
  const __m256i a1 =
      _mm256_loadu_si256((const __m256i *)(py + 32)); // cols  32..63
  const __m256i a2 =
      _mm256_loadu_si256((const __m256i *)(py + 64)); // cols  64..95
  const __m256i a3 =
      _mm256_loadu_si256((const __m256i *)(py + 96)); // cols 96..127

  // u8 (0..2) * s8, pairwise add -> int16
  const __m256i p0 = _mm256_maddubs_epi16(w0, a0);
  const __m256i p1 = _mm256_maddubs_epi16(w1, a1);
  const __m256i p2 = _mm256_maddubs_epi16(w2, a2);
  const __m256i p3 = _mm256_maddubs_epi16(w3, a3);

  return _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
}

// -------------------------------------------------------------------------------------
// INT4 helpers
// -------------------------------------------------------------------------------------

// Compute the offset sum of one 128-column block of INT4 activations.
// Each value s is mapped to s + 8 (in [1, 15]) and summed; returns 4 int64
// partial sums.
static inline __m256i sum_block_i4(
    const uint8_t
        *py, // Pointer to the block of the activation vector (INT4), 64 bytes
    const __m256i sign_flip,  // Constant vector to map both int4 nibbles to
                              // uint4 with offset 8 (0x88 per byte)
    const __m256i nibble_mask // Constant vector to mask out all bits except the
                              // first 4 (0x0F per byte)
) {
  const __m256i zero = _mm256_setzero_si256();

  const __m256i x0 =
      _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)py), sign_flip);
  const __m256i x1 = _mm256_xor_si256(
      _mm256_loadu_si256((const __m256i *)(py + 32)), sign_flip);

  // Low nibble + high nibble <= 30, so the byte addition cannot overflow
  const __m256i s0 =
      _mm256_add_epi8(_mm256_and_si256(x0, nibble_mask),
                      _mm256_and_si256(_mm256_srli_epi16(x0, 4), nibble_mask));
  const __m256i s1 =
      _mm256_add_epi8(_mm256_and_si256(x1, nibble_mask),
                      _mm256_and_si256(_mm256_srli_epi16(x1, 4), nibble_mask));

  return _mm256_add_epi64(_mm256_sad_epu8(s0, zero), _mm256_sad_epu8(s1, zero));
}

// Compute the dot product of one 128-column block between I2_S weights and INT4
// activations. Returns 16 int16 lanes; each lane is at most |4 * (2 * 7 + 2 *
// 7)| = 112.
static inline __m256i dot_block_i2_i4(
    const uint8_t
        *px, // Pointer to the block of the weight matrix (I2_S), 32 bytes
    const uint8_t
        *py, // Pointer to the block of the activation vector (INT4), 64 bytes
    const __m256i value_mask, // Constant vector to mask out all bits except the
                              // first 2 (0x03 per byte)
    const __m256i nibble_mask, // Constant vector to mask out all bits except
                               // the first 4 (0x0F per byte)
    const __m256i sign_lut // Lookup table to sign-extend a 4-bit nibble into an
                           // int8 value
) {
  // Unpack weights: 4 bit-planes -> unsigned 0/1/2
  __m256i w3 = _mm256_loadu_si256((const __m256i *)px);
  __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(w3, 2), value_mask);
  __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(w3, 4), value_mask);
  __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(w3, 6), value_mask);
  w3 = _mm256_and_si256(w3, value_mask);

  // Unpack activations: 2 values per byte -> sign-extended int8
  const __m256i x0 = _mm256_loadu_si256((const __m256i *)py);
  const __m256i x1 = _mm256_loadu_si256((const __m256i *)(py + 32));
  const __m256i a0 = _mm256_shuffle_epi8(
      sign_lut, _mm256_and_si256(_mm256_srli_epi16(x0, 4),
                                 nibble_mask)); // cols   0..31 (bits 7-4)
  const __m256i a1 = _mm256_shuffle_epi8(
      sign_lut, _mm256_and_si256(x0, nibble_mask)); // cols  32..63 (bits 3-0)
  const __m256i a2 = _mm256_shuffle_epi8(
      sign_lut, _mm256_and_si256(_mm256_srli_epi16(x1, 4),
                                 nibble_mask)); // cols  64..95 (bits 7-4)
  const __m256i a3 = _mm256_shuffle_epi8(
      sign_lut, _mm256_and_si256(x1, nibble_mask)); // cols 96..127 (bits 3-0)

  // u8 (0..2) * s8, pairwise add -> int16
  const __m256i p0 = _mm256_maddubs_epi16(w0, a0);
  const __m256i p1 = _mm256_maddubs_epi16(w1, a1);
  const __m256i p2 = _mm256_maddubs_epi16(w2, a2);
  const __m256i p3 = _mm256_maddubs_epi16(w3, a3);

  return _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
}

// =====================================================================================
// GEMV kernels
// =====================================================================================

// Perform GEMV with weight matrix in I2_S format and activation vector in INT8
// format
void gemv_i2_s_a8(
    const uint8_t
        *weight, // Weight matrix (I2_S), shape n_rows * ceil(n_cols / 128) * 32
    const int8_t *activation, // Activation vector (INT8), shape n_cols
    int32_t *output,          // Output vector (INT32), shape n_rows
    int n_rows,               // Number of rows of the weight matrix
    int n_cols // Number of columns of the weight matrix / size of the
               // activation vector
) {
  // 0. Initialize constants and variables
  const int COLS_PER_BLOCK =
      128; // Number of columns in each block (I2_S format)
  const int W_BYTES_PER_BLOCK =
      32; // Number of weight bytes in each block (128 x 2 bits)
  const int A_BYTES_PER_BLOCK =
      128; // Number of activation bytes in each block (128 x 8 bits)
  const int BLOCKS_PER_CHUNK =
      16; // Number of blocks accumulated in int16 before widening (16 x 2032 =
          // 32512 <= 32767)
  const int A_OFFSET =
      128; // Offset added to each activation value by sign_flip

  const int n_blocks_total = n_cols / COLS_PER_BLOCK; // Number of full blocks
  const int n_chunks =
      n_blocks_total / BLOCKS_PER_CHUNK; // Number of full chunks
  const int n_blocks_remaining =
      n_blocks_total %
      BLOCKS_PER_CHUNK; // Number of full blocks after the chunks
  const int n_cols_remaining =
      n_cols % COLS_PER_BLOCK; // Number of columns after the full blocks
  const int n_bytes_weight_row =
      (n_cols + 127) / 128 * 32; // Number of bytes per row of the weight matrix

  const __m256i one_mask =
      _mm256_set1_epi16(1); // Constant vector of ones for widening
  const __m256i value_mask =
      _mm256_set1_epi8(0x03); // Constant vector to keep the first 2 bits
  const __m256i sign_flip = _mm256_set1_epi8(
      (char)0x80); // Constant vector to map int8 to uint8 (+128)

  // 1. Copy the remaining columns of the activation vector into a zero-padded
  // block
  alignas(32) int8_t tail_activation[A_BYTES_PER_BLOCK] = {0};
  if (n_cols_remaining > 0) {
    const int n_bytes_tail = n_cols_remaining;
    memcpy(tail_activation,
           activation + (size_t)n_blocks_total * A_BYTES_PER_BLOCK,
           n_bytes_tail);
  }

  // 2. Compute the reduction sum of the activation vector
  // 2.1. Sum each full block
  __m256i sum_accumulator = _mm256_setzero_si256();
  for (int block_idx = 0; block_idx < n_blocks_total; ++block_idx) {
    sum_accumulator = _mm256_add_epi64(
        sum_accumulator,
        sum_block_i8(activation + (size_t)block_idx * A_BYTES_PER_BLOCK,
                     sign_flip));
  }

  // 2.2. Sum the zero-padded tail block (padding maps to A_OFFSET and is
  // removed in 2.3)
  if (n_cols_remaining > 0) {
    sum_accumulator = _mm256_add_epi64(
        sum_accumulator, sum_block_i8(tail_activation, sign_flip));
  }

  // 2.3. Reduce the partial sums and remove the offset
  const int64_t n_cols_processed =
      (int64_t)(n_blocks_total + (n_cols_remaining > 0 ? 1 : 0)) *
      COLS_PER_BLOCK;
  const int32_t activation_sum =
      (int32_t)(hsum_i64_4(sum_accumulator) - n_cols_processed * A_OFFSET);

  // 3. Perform GEMV operation for each row of the weight matrix
  for (int row = 0; row < n_rows; ++row) {
    // 3.0. Initialize the accumulator and the pointers to the current row
    __m256i accumulator = _mm256_setzero_si256();
    __m256i accumulator_chunk;
    const uint8_t *px = weight + (size_t)row * n_bytes_weight_row;
    const int8_t *py = activation;

    // 3.1. Process each chunk of BLOCKS_PER_CHUNK blocks
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
      accumulator_chunk = _mm256_setzero_si256();
      for (int block_idx = 0; block_idx < BLOCKS_PER_CHUNK; ++block_idx) {
        accumulator_chunk = _mm256_add_epi16(
            accumulator_chunk, dot_block_i2_i8(px, py, value_mask));
        px += W_BYTES_PER_BLOCK;
        py += A_BYTES_PER_BLOCK;
      }
      accumulator = _mm256_add_epi32(
          accumulator, _mm256_madd_epi16(accumulator_chunk, one_mask));
    }

    // 3.2. Process the remaining full blocks
    accumulator_chunk = _mm256_setzero_si256();
    for (int block_idx = 0; block_idx < n_blocks_remaining; ++block_idx) {
      accumulator_chunk = _mm256_add_epi16(accumulator_chunk,
                                           dot_block_i2_i8(px, py, value_mask));
      px += W_BYTES_PER_BLOCK;
      py += A_BYTES_PER_BLOCK;
    }

    // 3.3. Process the zero-padded tail block (the weight row is padded to 32
    // bytes, so the load is in bounds)
    if (n_cols_remaining > 0) {
      accumulator_chunk = _mm256_add_epi16(
          accumulator_chunk, dot_block_i2_i8(px, tail_activation, value_mask));
    }

    // 3.4. Widen the int16 accumulator of the last chunk into the int32
    // accumulator
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(accumulator_chunk, one_mask));

    // 3.5. Assign the final result: dot(W_u - 1, A) = dot(W_u, A) - sum(A)
    output[row] = hsum_i32_8(accumulator) - activation_sum;
  }
}

// Perform GEMV with weight matrix in I2_S format and activation vector in INT4
// format
void gemv_i2_s_a4(const uint8_t *weight, // Weight matrix (I2_S), shape n_rows *
                                         // ceil(n_cols / 128) * 32
                  const uint8_t *activation, // Activation vector (INT4), shape
                                             // ceil(n_cols / 64) * 32
                  int32_t *output, // Output vector (INT32), shape n_rows
                  int n_rows,      // Number of rows of the weight matrix
                  int n_cols // Number of columns of the weight matrix / size of
                             // the activation vector
) {
  // 0. Initialize constants and variables
  const int COLS_PER_BLOCK =
      128; // Number of columns in each block (I2_S format)
  const int W_BYTES_PER_BLOCK =
      32; // Number of weight bytes in each block (128 x 2 bits)
  const int A_BYTES_PER_BLOCK =
      64; // Number of activation bytes in each block (128 x 4 bits)
  const int BLOCKS_PER_CHUNK =
      256; // Number of blocks accumulated in int16 before widening (256 x 112 =
           // 28672 <= 32767)
  const int A_OFFSET = 8; // Offset added to each activation value by sign_flip

  const int n_blocks_total = n_cols / COLS_PER_BLOCK; // Number of full blocks
  const int n_chunks =
      n_blocks_total / BLOCKS_PER_CHUNK; // Number of full chunks
  const int n_blocks_remaining =
      n_blocks_total %
      BLOCKS_PER_CHUNK; // Number of full blocks after the chunks
  const int n_cols_remaining =
      n_cols % COLS_PER_BLOCK; // Number of columns after the full blocks
  const int n_bytes_weight_row =
      (n_cols + 127) / 128 * 32; // Number of bytes per row of the weight matrix

  const __m256i one_mask =
      _mm256_set1_epi16(1); // Constant vector of ones for widening
  const __m256i value_mask =
      _mm256_set1_epi8(0x03); // Constant vector to keep the first 2 bits
  const __m256i sign_flip = _mm256_set1_epi8(
      (char)0x88); // Constant vector to map both int4 nibbles to uint4 (+8)
  const __m256i nibble_mask =
      _mm256_set1_epi8(0x0F); // Constant vector to keep the first 4 bits
  const __m256i sign_lut =
      _mm256_setr_epi8( // Lookup table to sign-extend a nibble into int8
          0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
          5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);

  // 1. Copy the remaining columns of the activation vector into a zero-padded
  // block
  alignas(32) uint8_t tail_activation[A_BYTES_PER_BLOCK] = {0};
  if (n_cols_remaining > 0) {
    const int n_bytes_tail =
        (n_cols_remaining + 63) / 64 *
        32; // Whole groups of 64 values (A[i] and A[i + 32] share a byte)
    memcpy(tail_activation,
           activation + (size_t)n_blocks_total * A_BYTES_PER_BLOCK,
           n_bytes_tail);
  }

  // 2. Compute the reduction sum of the activation vector
  // 2.1. Sum each full block
  __m256i sum_accumulator = _mm256_setzero_si256();
  for (int block_idx = 0; block_idx < n_blocks_total; ++block_idx) {
    sum_accumulator = _mm256_add_epi64(
        sum_accumulator,
        sum_block_i4(activation + (size_t)block_idx * A_BYTES_PER_BLOCK,
                     sign_flip, nibble_mask));
  }

  // 2.2. Sum the zero-padded tail block (padding maps to A_OFFSET and is
  // removed in 2.3)
  if (n_cols_remaining > 0) {
    sum_accumulator = _mm256_add_epi64(
        sum_accumulator, sum_block_i4(tail_activation, sign_flip, nibble_mask));
  }

  // 2.3. Reduce the partial sums and remove the offset
  const int64_t n_cols_processed =
      (int64_t)(n_blocks_total + (n_cols_remaining > 0 ? 1 : 0)) *
      COLS_PER_BLOCK;
  const int32_t activation_sum =
      (int32_t)(hsum_i64_4(sum_accumulator) - n_cols_processed * A_OFFSET);

  // 3. Perform GEMV operation for each row of the weight matrix
  for (int row = 0; row < n_rows; ++row) {
    // 3.0. Initialize the accumulator and the pointers to the current row
    __m256i accumulator = _mm256_setzero_si256();
    __m256i accumulator_chunk;
    const uint8_t *px = weight + (size_t)row * n_bytes_weight_row;
    const uint8_t *py = activation;

    // 3.1. Process each chunk of BLOCKS_PER_CHUNK blocks
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
      accumulator_chunk = _mm256_setzero_si256();
      for (int block_idx = 0; block_idx < BLOCKS_PER_CHUNK; ++block_idx) {
        accumulator_chunk = _mm256_add_epi16(
            accumulator_chunk,
            dot_block_i2_i4(px, py, value_mask, nibble_mask, sign_lut));
        px += W_BYTES_PER_BLOCK;
        py += A_BYTES_PER_BLOCK;
      }
      accumulator = _mm256_add_epi32(
          accumulator, _mm256_madd_epi16(accumulator_chunk, one_mask));
    }

    // 3.2. Process the remaining full blocks
    accumulator_chunk = _mm256_setzero_si256();
    for (int block_idx = 0; block_idx < n_blocks_remaining; ++block_idx) {
      accumulator_chunk = _mm256_add_epi16(
          accumulator_chunk,
          dot_block_i2_i4(px, py, value_mask, nibble_mask, sign_lut));
      px += W_BYTES_PER_BLOCK;
      py += A_BYTES_PER_BLOCK;
    }

    // 3.3. Process the zero-padded tail block (the weight row is padded to 32
    // bytes, so the load is in bounds)
    if (n_cols_remaining > 0) {
      accumulator_chunk = _mm256_add_epi16(
          accumulator_chunk, dot_block_i2_i4(px, tail_activation, value_mask,
                                             nibble_mask, sign_lut));
    }

    // 3.4. Widen the int16 accumulator of the last chunk into the int32
    // accumulator
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(accumulator_chunk, one_mask));

    // 3.5. Assign the final result: dot(W_u - 1, A) = dot(W_u, A) - sum(A)
    output[row] = hsum_i32_8(accumulator) - activation_sum;
  }
}