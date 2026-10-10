import numpy as np

def compute_gemv_i2_s_a4(
    weight: np.ndarray, 
    activation: np.ndarray, 
    n_rows: int, 
    n_cols: int
) -> np.ndarray: ...

def compute_gemv_i2_s_a8(
    weight: np.ndarray, 
    activation: np.ndarray, 
    n_rows: int, 
    n_cols: int
) -> np.ndarray: ...