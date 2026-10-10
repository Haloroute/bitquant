#pragma once
#include <cstdint>

// Function to perform GEMV operation with int2 weights and int8 activations
void gemv_i2_s_a8(
    const uint8_t *weight,    // Weight matrix W (I2_S format) (shape n_rows *
                              // ceil(n_cols / 128) * 32)
    const int8_t *activation, // Activation vector A (shape n_cols)
    int32_t *output,          // Output vector (shape n_rows)
    int n_rows,               // Number of rows in the weight matrix
    int n_cols // Number of columns in the weight matrix and size of the
               // activation vector
);

// Function to perform GEMV operation with int2 weights and int4 activations
void gemv_i2_s_a4(
    const uint8_t *weight,     // Weight matrix W (I2_S format) (shape n_rows *
                               // ceil(n_cols / 128) * 32)
    const uint8_t *activation, // Activation vector A (shape n_cols)
    int32_t *output,           // Output vector (shape n_rows)
    int n_rows,                // Number of rows in the weight matrix
    int n_cols // Number of columns in the weight matrix and size of the
               // activation vector
);