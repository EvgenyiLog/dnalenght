import numpy as np
from numpy.typing import NDArray


def load_matlab_data(filename: str) -> NDArray:
    print(filename)
    with open(filename, "r") as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


def compare_data(data1: NDArray, data2: NDArray) -> None:
    delta = data1 - data2
    # Mean Squared Error (MSE)
    mse = np.mean((delta) ** 2)
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(delta))
    # Maximum absolute difference
    max_diff = np.max(np.abs(delta))
    # Pearson correlation coefficient
    corr_coef = np.corrcoef(delta)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"Max absolute difference: {max_diff}")
    print(f"Correlation coefficient: {corr_coef}")


def compare_with_file(data: NDArray, filename: str) -> None:
    return compare_data(data, load_matlab_data(filename))
