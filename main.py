import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1234
NUM_SAMPLES = 50
np.random.seed(SEED)


def generate_data(num_samples):
    x = np.array(range(num_samples))
    noise = np.random.uniform(-10, 20, size=num_samples)
    y = 3.5 * x + noise
    return x, y


x, y = generate_data(num_samples=NUM_SAMPLES)
data = np.vstack([x, y]).T
