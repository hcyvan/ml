import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam

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
###########################################################################################
###########################################################################################
###########################################################################################
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
print("[actual] y = 3.5x + noise")
print(f"[model] y_hat = {W_unscaled[0][0]:.1f} x + {b_unscaled[0][0]:.1f}")

###########################################################################################
###########################################################################################
###########################################################################################
torch.manual_seed(SEED)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SHUFFLE = True


def train_val_test_split(x, y, val_size, test_size, shuffle):
    """Split data into train/val/test datasets.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=shuffle)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, shuffle=shuffle)
    return x_train, x_val, x_test, y_train, y_val, y_test


x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
    x, y, val_size=VAL_SIZE, test_size=TEST_SIZE, shuffle=SHUFFLE)
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_val: {x_val.shape}, y_test: {y_val.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

x_scaler = StandardScaler().fit(x_train)
y_scaler = StandardScaler().fit(y_train)
x_train = x_scaler.transform(x_train)
y_train = y_scaler.transform(y_train).ravel().reshape(-1, 1)
x_val = x_scaler.transform(x_val)
y_val = y_scaler.transform(y_val).ravel().reshape(-1, 1)
x_test = x_scaler.transform(x_test)
y_test = y_scaler.transform(y_test).ravel().reshape(-1, 1)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_val = torch.Tensor(x_val)
y_val = torch.Tensor(y_val)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x_in):
        y_pred = self.fc1(x_in)
        return y_pred


L2_LAMBDA = 1e-2
model = LinearRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

for epoch in range(100):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | loss: {loss:.2f}")

pred_train = model(x_train)
pred_test = model(x_test)
train_error = loss_fn(pred_train, y_train)
test_error = loss_fn(pred_test, y_test)
print(f'train_error: {train_error:.2f}')
print(f'test_error: {test_error:.2f}')

# Figure size
plt.figure(figsize=(15, 5))

# Plot train data
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(x_train, y_train, label='y_train')
plt.plot(x_train, pred_train.detach().numpy(), color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')

# Plot test data
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(x_test, y_test, label='y_test')
plt.plot(x_test, pred_test.detach().numpy(), color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')

# Show plots
plt.show()

sample_indices = [10, 15, 25]
x_infer = np.array(sample_indices, dtype=np.float32)
x_infer = torch.Tensor(x_scaler.transform(x_infer.reshape(-1, 1)))
pred_infer = model(x_infer).detach().numpy() * np.sqrt(y_scaler.var_) + y_scaler.mean_
for i, index in enumerate(sample_indices):
    print(f"{df.iloc[index]['y']:.2f} (actual) → {pred_infer[i][0]:.2f} (predicted)")
