import numpy as np


def normalize_array(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return (arr - mean) / std


def filter_outliers(arr: np.ndarray, threshold: float) -> np.ndarray:
    mean = np.mean(arr)
    std = np.std(arr)
    mask = np.abs(arr - mean) <= threshold * std
    return arr[mask]