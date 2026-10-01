import numpy as np

def compute_gemv(
    weight: np.ndarray, 
    activation: np.ndarray, 
    n_rows: int, 
    n_cols: int
) -> np.ndarray: ...