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

df = pd.DataFrame(data, columns=['x', 'y'])
x = df[['x']].values
y = df[['y']].values

plt.title("Generate data")
plt.scatter(x=df['x'], y=df['y'])
plt.show()

# shuffle
indices = list(range(NUM_SAMPLES))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

train_start = 0
train_end = int(0.7 * NUM_SAMPLES)
val_start = train_end
val_end = int((TRAIN_SIZE + VAL_SIZE) * NUM_SAMPLES)
test_start = val_end

print(train_start, train_end)
print(val_start, val_end)
print(test_start, NUM_SAMPLES)

x_train = x[train_start:train_end]
y_train = y[train_start:train_end]
x_val = x[val_start:val_end]
y_val = y[val_start:val_end]
x_test = x[test_start:]
y_test = y[test_start:]
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_val: {x_val.shape}, y_test: {y_val.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

plt.scatter(x_train, y_train)
plt.show()
plt.scatter(x_val, y_val)
plt.show()
plt.scatter(x_test, y_test)
plt.show()


def standardize_data(data, mean, std):
    return (data - mean) / std


x_mean = np.mean(x_train)
x_std = np.std(x_train)
y_mean = np.mean(y_train)
y_std = np.std(y_train)

x_train = standardize_data(x_train, x_mean, x_std)
y_train = standardize_data(y_train, y_mean, y_std)
x_val = standardize_data(x_val, x_mean, x_std)
y_val = standardize_data(y_val, y_mean, y_std)
x_test = standardize_data(x_test, x_mean, x_std)
y_test = standardize_data(y_test, y_mean, y_std)

INPUT_DIM = x_train.shape[1]
OUTPUT_DIM = y_train.shape[1]

W = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
b = np.zeros((1, 1))

N = len(y_train)
LEARNING_RATE = 1e-1
for epoch_num in range(100):
    # Forward
    y_pred = np.dot(x_train, W) + b
    loss = (1 / N) * np.sum((y_train - y_pred) ** 2)
    if epoch_num % 10 == 0:
        print(f"Epoch: {epoch_num}, loss: {loss:.3f}")
    # Backpropagation
    dW = -(2 / N) * np.sum((y_train - y_pred) * x_train)
    db = -(2 / N) * np.sum((y_train - y_pred) * 1)
    # Update weights
    W += -LEARNING_RATE * dW
    b += -LEARNING_RATE * db

pred_train = W * x_train + b
pred_test = W * x_test + b

train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print(f"train_MSE: {train_mse:.2f}, test_MSE: {test_mse:.2f}")

# Figure size
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(x_train, y_train, label='y_train')
plt.plot(x_train, pred_train, color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(x_test, y_test, label='y_test')
plt.plot(x_test, pred_test, color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')
plt.show()

# Unscaled weights
W_unscaled = W * (y_std / x_std)
b_unscaled = b * y_std + y_mean - np.sum(W_unscaled * x_mean)
print("[actual] y = 3.5X + noise")
print(f"[model] y_hat = {W_unscaled[0][0]:.1f} X + {b_unscaled[0][0]:.1f}")
