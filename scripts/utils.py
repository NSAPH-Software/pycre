import numpy as np

def standardize(array):
    mu = np.mean(array)
    std = np.std(array)
    return (array-mu) / std
