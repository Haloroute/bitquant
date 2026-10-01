#ifndef GEMV_H
#define GEMV_H
#include <stdint.h>

void gemv_i2_s(const uint8_t *weight, const int8_t *activation, int32_t *output,
               int n_rows, int n_cols);
#endif