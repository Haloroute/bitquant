#pragma once
#include <cstdint>

// Function to pack a INT4 vector in INT8 format into A4 format
void pack_1d_int4(const int8_t *activation, uint8_t *output, int n_cols);

// Function to pack a 1.58-bit matrix in INT8 format into I2_S format
void pack_2d_i2s(const int8_t *weight, uint8_t *output, int n_rows, int n_cols);