#include "gemv.h"
#include "immintrin.h"
#include <cstring>

// Helper function to horizontally sum 8 int32 values in a __m256i vector
static inline int hsum_i32_8(const __m256i a) {
  const __m128i sum128 =
      _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
  const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
  const __m128i sum64 = _mm_add_epi32(hi64, sum128);
  const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Perform GEMV operation with weight matrix in I2_S format and activation
// vector in INT8 format
void gemv_i2_s(const uint8_t *weight,    // Weight matrix W (I2_S format) (shape
                                         // n_rows * ceil(n_cols / 128) * 32)
               const int8_t *activation, // Activation vector A (shape n_cols)
               int32_t *output,          // Output vector (shape n_rows)
               int n_rows,               // Number of rows in the weight matrix
               int n_cols // Number of columns in the weight matrix and size of
                          // the activation vector
) {
  // 0. Initialize constants and variables
  const int COLS_PER_BLOCK =
      128; // Number of columns processed in each block (I2_S format)
  const int BLOCKS_PER_CHUNK =
      32; // Number of blocks processed in each chunk (I2_S format)

  const int n_chunks =
      n_cols /
      (COLS_PER_BLOCK * BLOCKS_PER_CHUNK); // Number of chunks to process
  const int n_blocks_remaining =
      (n_cols % (COLS_PER_BLOCK * BLOCKS_PER_CHUNK)) /
      COLS_PER_BLOCK; // Number of remaining blocks after chunks
  const int n_cols_remaining =
      n_cols % COLS_PER_BLOCK; // Number of remaining columns after blocks
  const int n_cols_weight =
      (n_cols + 127) / 128 *
      32; // Number of bytes per row in the weight matrix (I2_S format)

  const __m256i one_mask =
      _mm256_set1_epi16(1); // Constant vector of ones for reduction
  const __m256i value_mask = _mm256_set1_epi8(
      0x03); // Constant vector to mask out all bits except the first 2

  // 1. Compute the reduction sum of activation vector
  // Khởi tạo mảng đệm 128 phần tử (toàn 0) dùng chung cho tổng và GEMV
  alignas(32) int8_t tail_activation[128] = {0};
  if (n_cols_remaining > 0) {
    memcpy(tail_activation, activation + n_cols - n_cols_remaining,
           n_cols_remaining);
  }

  const __m256i v_offset = _mm256_set1_epi8(-128); // 0x80
  const __m256i v_zero = _mm256_setzero_si256();
  __m256i acc0 = v_zero, acc1 = v_zero, acc2 = v_zero, acc3 = v_zero;

  const int n_blocks_total = n_cols / COLS_PER_BLOCK;

  // 1.1 Tính tổng các khối 128 phần tử trọn vẹn
  for (int i = 0; i < n_blocks_total; ++i) {
    const int8_t *ptr = activation + i * COLS_PER_BLOCK;
    __m256i v0 =
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)ptr), v_offset);
    __m256i v1 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(ptr + 32)), v_offset);
    __m256i v2 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(ptr + 64)), v_offset);
    __m256i v3 = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)(ptr + 96)), v_offset);

    acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(v0, v_zero));
    acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(v1, v_zero));
    acc2 = _mm256_add_epi64(acc2, _mm256_sad_epu8(v2, v_zero));
    acc3 = _mm256_add_epi64(acc3, _mm256_sad_epu8(v3, v_zero));
  }

  // 1.2 Tính tổng phần dư bằng mảng pad 0
  if (n_cols_remaining > 0) {
    __m256i v0 = _mm256_xor_si256(
        _mm256_load_si256((const __m256i *)tail_activation), v_offset);
    __m256i v1 = _mm256_xor_si256(
        _mm256_load_si256((const __m256i *)(tail_activation + 32)), v_offset);
    __m256i v2 = _mm256_xor_si256(
        _mm256_load_si256((const __m256i *)(tail_activation + 64)), v_offset);
    __m256i v3 = _mm256_xor_si256(
        _mm256_load_si256((const __m256i *)(tail_activation + 96)), v_offset);

    acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(v0, v_zero));
    acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(v1, v_zero));
    acc2 = _mm256_add_epi64(acc2, _mm256_sad_epu8(v2, v_zero));
    acc3 = _mm256_add_epi64(acc3, _mm256_sad_epu8(v3, v_zero));
  }

  // 1.3 Gom kết quả và trừ đi bias
  __m256i sum256 = _mm256_add_epi64(_mm256_add_epi64(acc0, acc1),
                                    _mm256_add_epi64(acc2, acc3));
  alignas(32) int64_t vals[4];
  _mm256_store_si256((__m256i *)vals, sum256);
  int64_t total_unsigned = vals[0] + vals[1] + vals[2] + vals[3];

  int total_processed =
      (n_blocks_total + (n_cols_remaining > 0 ? 1 : 0)) * COLS_PER_BLOCK;
  int32_t activation_sum =
      static_cast<int32_t>(total_unsigned - total_processed * 128);

  // 2. Perform GEMV operation for each row of the weight matrix
  for (int row = 0; row < n_rows; ++row) {
    // 2.0. Initialize accumulator for the dot product
    __m256i accumulator = _mm256_setzero_si256(),
            accumulator_chunk = _mm256_setzero_si256();
    const uint8_t *px_row = weight + row * n_cols_weight,
                  *px; // Pointer to the current row in the weight matrix
    const int8_t *py;

    // 2.1. Process each chunk of 4096 columns (32 blocks of 128 columns)
    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
      // 2.1.0. Initialize pointers to the current chunk in the weight matrix
      // and activation vector
      px = px_row + chunk_idx * BLOCKS_PER_CHUNK * COLS_PER_BLOCK /
                        4; // Pointer to the current chunk in the weight matrix
      py = activation + chunk_idx * BLOCKS_PER_CHUNK *
                            COLS_PER_BLOCK; // Pointer to the current chunk in
                                            // the activation vector
      accumulator_chunk =
          _mm256_setzero_si256(); // Accumulator for the current chunk

      // 2.1.1. Process each block in the current chunk
      for (int block_idx = 0; block_idx < BLOCKS_PER_CHUNK; ++block_idx) {
        // 2.1.1.0. Load 128 weights and activation into 8 AVX2 register (32
        // each)
        __m256i weight_vec_3 = _mm256_loadu_si256((const __m256i *)px);
        __m256i weight_vec_2 = _mm256_srli_epi16(weight_vec_3, 2);
        __m256i weight_vec_1 = _mm256_srli_epi16(weight_vec_3, 4);
        __m256i weight_vec_0 = _mm256_srli_epi16(weight_vec_3, 6);

        __m256i activation_vec_0 = _mm256_loadu_si256((const __m256i *)py);
        __m256i activation_vec_1 =
            _mm256_loadu_si256((const __m256i *)(py + 32));
        __m256i activation_vec_2 =
            _mm256_loadu_si256((const __m256i *)(py + 64));
        __m256i activation_vec_3 =
            _mm256_loadu_si256((const __m256i *)(py + 96));

        // 2.1.1.1. Pad weights with suitable mask
        weight_vec_3 = _mm256_and_si256(weight_vec_3, value_mask);
        weight_vec_2 = _mm256_and_si256(weight_vec_2, value_mask);
        weight_vec_1 = _mm256_and_si256(weight_vec_1, value_mask);
        weight_vec_0 = _mm256_and_si256(weight_vec_0, value_mask);

        // 2.1.1.2. Calculate dot product
        weight_vec_0 = _mm256_maddubs_epi16(weight_vec_0, activation_vec_0);
        weight_vec_1 = _mm256_maddubs_epi16(weight_vec_1, activation_vec_1);
        weight_vec_2 = _mm256_maddubs_epi16(weight_vec_2, activation_vec_2);
        weight_vec_3 = _mm256_maddubs_epi16(weight_vec_3, activation_vec_3);

        // 2.1.1.3. Accumulate into accumulator
        accumulator_chunk = _mm256_add_epi16(
            accumulator_chunk, _mm256_add_epi16(weight_vec_0, weight_vec_1));
        accumulator_chunk = _mm256_add_epi16(
            accumulator_chunk, _mm256_add_epi16(weight_vec_2, weight_vec_3));

        // 2.1.1.4. Update pointers
        px += COLS_PER_BLOCK / 4;
        py += COLS_PER_BLOCK;
      }

      // 2.1.2. Accumulate 16-bit accumulator into 32-bit one
      accumulator = _mm256_add_epi32(
          accumulator, _mm256_madd_epi16(accumulator_chunk, one_mask));
    }

    // 2.2. Process each remaining block of 128 columns
    // 2.2.0. Initialize pointers to the current chunk in the weight matrix and
    // activation vector
    px = px_row + n_chunks * BLOCKS_PER_CHUNK * COLS_PER_BLOCK /
                      4; // Pointer to the current chunk in the weight matrix
    py = activation + n_chunks * BLOCKS_PER_CHUNK *
                          COLS_PER_BLOCK; // Pointer to the current chunk in the
                                          // activation vector
    accumulator_chunk =
        _mm256_setzero_si256(); // Accumulator for the current chunk

    // 2.2.1. Process each block in the current chunk
    for (int block_idx = 0; block_idx < n_blocks_remaining; ++block_idx) {
      // 2.1.1.0. Load 128 weights and activation into 8 AVX2 register (32 each)
      __m256i weight_vec_3 = _mm256_loadu_si256((const __m256i *)px);
      __m256i weight_vec_2 = _mm256_srli_epi16(weight_vec_3, 2);
      __m256i weight_vec_1 = _mm256_srli_epi16(weight_vec_3, 4);
      __m256i weight_vec_0 = _mm256_srli_epi16(weight_vec_3, 6);

      __m256i activation_vec_0 = _mm256_loadu_si256((const __m256i *)py);
      __m256i activation_vec_1 = _mm256_loadu_si256((const __m256i *)(py + 32));
      __m256i activation_vec_2 = _mm256_loadu_si256((const __m256i *)(py + 64));
      __m256i activation_vec_3 = _mm256_loadu_si256((const __m256i *)(py + 96));

      // 2.1.1.1. Pad weights with suitable mask
      weight_vec_3 = _mm256_and_si256(weight_vec_3, value_mask);
      weight_vec_2 = _mm256_and_si256(weight_vec_2, value_mask);
      weight_vec_1 = _mm256_and_si256(weight_vec_1, value_mask);
      weight_vec_0 = _mm256_and_si256(weight_vec_0, value_mask);

      // 2.1.1.2. Calculate dot product
      weight_vec_0 = _mm256_maddubs_epi16(weight_vec_0, activation_vec_0);
      weight_vec_1 = _mm256_maddubs_epi16(weight_vec_1, activation_vec_1);
      weight_vec_2 = _mm256_maddubs_epi16(weight_vec_2, activation_vec_2);
      weight_vec_3 = _mm256_maddubs_epi16(weight_vec_3, activation_vec_3);

      // 2.1.1.3. Accumulate into accumulator
      accumulator_chunk = _mm256_add_epi16(
          accumulator_chunk, _mm256_add_epi16(weight_vec_0, weight_vec_1));
      accumulator_chunk = _mm256_add_epi16(
          accumulator_chunk, _mm256_add_epi16(weight_vec_2, weight_vec_3));

      // 2.1.1.4. Update pointers
      px += COLS_PER_BLOCK / 4;
      py += COLS_PER_BLOCK;
    }

    // 2.2.2. Accumulate 16-bit accumulator into 32-bit one
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(accumulator_chunk, one_mask));

    // 2.3. Process any remaining columns (less than 128) using SIMD
    if (n_cols_remaining > 0) {
      // Sử dụng mảng tail_activation đã được pad 0 làm đầu vào
      const int8_t *py_tail = tail_activation;

      // Load 128 weights (32 bytes) - An toàn vì n_cols_weight đã được làm tròn
      __m256i weight_vec_3 = _mm256_loadu_si256((const __m256i *)px);
      __m256i weight_vec_2 = _mm256_srli_epi16(weight_vec_3, 2);
      __m256i weight_vec_1 = _mm256_srli_epi16(weight_vec_3, 4);
      __m256i weight_vec_0 = _mm256_srli_epi16(weight_vec_3, 6);

      __m256i activation_vec_0 = _mm256_loadu_si256((const __m256i *)py_tail);
      __m256i activation_vec_1 =
          _mm256_loadu_si256((const __m256i *)(py_tail + 32));
      __m256i activation_vec_2 =
          _mm256_loadu_si256((const __m256i *)(py_tail + 64));
      __m256i activation_vec_3 =
          _mm256_loadu_si256((const __m256i *)(py_tail + 96));

      weight_vec_3 = _mm256_and_si256(weight_vec_3, value_mask);
      weight_vec_2 = _mm256_and_si256(weight_vec_2, value_mask);
      weight_vec_1 = _mm256_and_si256(weight_vec_1, value_mask);
      weight_vec_0 = _mm256_and_si256(weight_vec_0, value_mask);

      weight_vec_0 = _mm256_maddubs_epi16(weight_vec_0, activation_vec_0);
      weight_vec_1 = _mm256_maddubs_epi16(weight_vec_1, activation_vec_1);
      weight_vec_2 = _mm256_maddubs_epi16(weight_vec_2, activation_vec_2);
      weight_vec_3 = _mm256_maddubs_epi16(weight_vec_3, activation_vec_3);

      __m256i accumulator_tail = _mm256_setzero_si256();
      accumulator_tail = _mm256_add_epi16(
          accumulator_tail, _mm256_add_epi16(weight_vec_0, weight_vec_1));
      accumulator_tail = _mm256_add_epi16(
          accumulator_tail, _mm256_add_epi16(weight_vec_2, weight_vec_3));

      // Cộng dồn trực tiếp vào accumulator 32-bit chính
      accumulator = _mm256_add_epi32(
          accumulator, _mm256_madd_epi16(accumulator_tail, one_mask));
    }

    // Assign final result to output vector
    output[row] = hsum_i32_8(accumulator) - activation_sum;
  }
}