# Stock price prediction with a two-layer RNN in PyTorch, replicating the setup
# from the research paper. Data: daily AAPL prices, downloaded by fetch_aapl.py
# into aapl_prices.csv.
#
# Libraries: pandas for loading and handling the data table,
# sklearn for scaling values and computing metrics, matplotlib for drawing the chart
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the AAPL price history from a CSV file. The Date column becomes the index.
df = pd.read_csv("aapl_prices.csv", index_col="Date", parse_dates=True)
df = df.sort_values("Date")  # Sort the data by date (oldest first)

# Checking for NaN values in the "Open" column and dropping them if any exist
if df["Open"].isnull().any():
    print("NaN values found in 'Open' column. Dropping rows with NaN values.")
    df = df.dropna(subset=["Open"])

# Optional plot of the raw Open prices (kept in a string so it doesn't run; remove the quotes to show it)

'''
plt.figure(figsize=(8, 6))
plt.plot(df.index, df["Open"], label="Open")
plt.title("AAPL Open Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''


TIMESTEPS = 5   # The number of previous days to use as input for predicting the next day's price
TRAIN_TEST_SPLIT = 0.65  # The fraction of the data to use for training (the rest is for testing)

# Row position where the test period starts: the earlier rows are for training, the rest for testing
split_index = int(len(df) * TRAIN_TEST_SPLIT)
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

# print(f"Training data: {len(train_data)} rows, Testing data: {len(test_data)} rows")

# Scale the "Open" prices to the range [0, 1] using MinMaxScaler.
# The scaler is fit on the training data only, so no information from the test period leaks into training.
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[["Open"]])
test_scaled = scaler.transform(test_data[["Open"]])

#print(f"Scaled training data shape: {train_scaled.shape}, Scaled testing data shape: {test_scaled.shape}")
#print(type(train_scaled), type(test_scaled))  # Check the types of the scaled data


# Turn the scaled prices into (input window, target) pairs. Each input is TIMESTEPS consecutive
# prices, and its target is the price on the day after the window.
def create_sequences(data, timesteps):
    inputs = []
    targets = []
    for i in range(len(data) - timesteps):
        inputs.append(data[i:i + timesteps])
        targets.append(data[i + timesteps])
    return np.array(inputs), np.array(targets)

# Build the windows for each period separately. The first TIMESTEPS test days only serve as inputs,
# so the first test prediction is for the sixth test day.
X_train, y_train = create_sequences(train_scaled, TIMESTEPS)
X_test, y_test = create_sequences(test_scaled, TIMESTEPS)

# Hold out the last 10% of the training windows as a validation set.
# Keep the time order: no shuffling before the cut.
VAL_SPLIT = 0.9
val_index = int(len(X_train) * VAL_SPLIT)
X_val, y_val = X_train[val_index:], y_train[val_index:]
X_train, y_train = X_train[:val_index], y_train[:val_index]

#print(f"Train windows: {len(X_train)}, Validation windows: {len(X_val)}, Test windows: {len(X_test)}")

# print("X_train[0] (input):\n", X_train[0])
# print("y_train[0] (target):", y_train[0])

# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
# print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# print(X_train.dtype, y_train.dtype, X_test.dtype, y_test.dtype, X_val.dtype, y_val.dtype)  # Check the data types of the arrays

# Convert the NumPy arrays to PyTorch tensors, which the model needs
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# print(X_train.dtype, y_train.dtype, X_test.dtype, y_test.dtype, X_val.dtype, y_val.dtype)  # Check the data types of the arrays

# Pair each input window with its target, then group them into batches of 64 windows for the model
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle the training data for better generalization

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # No shuffling for testing for consistent evaluation

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # No shuffling for validation for consistent evaluation

# Fix the random seeds so the weights, batch order and dropout are the same on every run
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Two stacked RNN layers (50 units, then 100 units) with dropout between them,
# followed by a linear layer that outputs one predicted price. Matches the paper's architecture.
class StockPriceRNN(nn.Module):
    def __init__(self, input_size=1, units1=50, units2=100, output_size=1, dropout=0.2):
        super(StockPriceRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, units1, batch_first=True)
        self.rnn2 = nn.RNN(units1, units2, batch_first=True)
        self.fc = nn.Linear(units2, output_size)
        self.dropout = nn.Dropout(dropout)

    # Pass the input through: RNN layer 1 -> dropout -> RNN layer 2 -> last time step -> linear layer
    def forward(self, x):
        out1, _ = self.rnn1(x)
        out1 = self.dropout(out1)
        out2, _ = self.rnn2(out1)
        last = out2[:, -1, :]  # store the output of the last time step to feed into the fully connected layer
        out = self.fc(last)   # Computes weighted sum of the last time step's output and adds a bias term, producing the final prediction for the next day's price
        return out

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available, otherwise fallback to CPU
model = StockPriceRNN(input_size=1, units1=50, units2=100, output_size=1, dropout=0.2).to(device)

criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimiser = optim.Adam(model.parameters())  # Adam optimiser for training the model with default learning rate of 0.001

EPOCHS = 50  # Number of epochs to train the model

# Loss history for each epoch, kept so the curves can be plotted after training
train_losses = []
val_losses = []

# One epoch is one full pass over the training data, followed by a check on the validation data
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    epoch_train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        optimiser.zero_grad()  # Clear the gradients before backpropagation
        loss.backward()  # Backpropagation to compute gradients
        optimiser.step()  # Update the model parameters based on the computed gradients
        epoch_train_loss += loss.item() * len(inputs) # Accumulate the loss for the epoch

    epoch_train_loss /= len(train_loader)  # Average loss for the epoch

    model.eval()  # Set the model to evaluation mode
    epoch_val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            epoch_val_loss += loss.item() * len(inputs)  # Accumulate the loss for the epoch

    epoch_val_loss /= len(val_loader)  # Average loss for the epoch

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}")


'''
# Plotting the training and validation losses over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Training loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation loss")
plt.title("Training and Validation Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE (scaled)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

'''

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
test_predictions = []
with torch.no_grad():  # Disable gradient computation for testing
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        predictions = model(inputs)
        test_predictions.append(predictions.cpu().numpy())  # Move to CPU and convert to NumPy for scikit-learn

test_preds = scaler.inverse_transform(np.concatenate(test_predictions, axis=0))  # Inverse transform to get the original scale of prices
test_actuals = scaler.inverse_transform(y_test.cpu().numpy())  # Inverse transform the actual test targets

# Calculate evaluation metrics
mse = mean_squared_error(test_actuals, test_preds)
mae = mean_absolute_error(test_actuals, test_preds)
rmse = np.sqrt(mse)

print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")


# Plotting the actual vs predicted prices for the test set
test_dates = df.index[split_index + TIMESTEPS : split_index + TIMESTEPS + len(test_actuals)]  # Get the corresponding dates for the test set

plt.figure(figsize=(10, 6))
plt.plot(test_dates, test_actuals, label="Actual Open price")
plt.plot(test_dates, test_preds, label="Predicted Open price")
plt.title("AAPL Open Price: Actual vs Predicted (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
