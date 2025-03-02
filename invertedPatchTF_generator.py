import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from torch.optim.lr_scheduler import LambdaLR

# Define the learning rate warmup function
def lr_lambda(epoch):
    warmup_epochs = 10  # Warmup for 10 epochs
    return min((epoch + 1) ** 0.5, 1) if epoch < warmup_epochs else 1

input_length = 96      # Input sequence length
output_length = 720    # Output sequence length
patch_length = 12      # Length of each patch
num_patches = input_length // patch_length  # Number of patches

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Convert date to datetime (if applicable)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')  # Sort by date
        data = data.drop(columns=['date'])  # Drop the date column
    return data.values  # Convert to numpy array

# Normalize data
def normalize_data(train, val, test):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    return train, val, test, scaler

# Create sequences for input and output
def create_sequences(data, input_length, output_length):
    inputs, outputs = [], []
    for i in range(len(data) - input_length - output_length + 1):
        inputs.append(data[i:i + input_length])
        outputs.append(data[i + input_length:i + input_length + output_length, -1])  # Target is Solar Power Output
    return np.array(inputs), np.array(outputs)

# Patch the time series
def patchify(data, patch_length, num_patches):
    patches = []
    for i in range(num_patches):
        start = i * patch_length
        end = start + patch_length
        patches.append(data[:, start:end, :])
    return np.stack(patches, axis=1)

# Inverted Transformer Layer
class InvertedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(InvertedTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Reverse the sequence for inverted attention
        src_reversed = torch.flip(src, dims=[0])
        # Apply self-attention
        src2, _ = self.self_attn(src_reversed, src_reversed, src_reversed)
        # Add & Norm
        src = src + self.dropout(src2)
        src = self.norm1(src)
        # Feedforward network
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

# Inverted PatchTST Model
class InvertedPatchTST(nn.Module):
    def __init__(self, patch_length, num_patches, d_model, nhead, num_layers, dropout, num_features):
        super(InvertedPatchTST, self).__init__()
        self.patch_length = patch_length
        self.num_patches = num_patches
        self.d_model = d_model

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_length * num_features, d_model)

        # Inverted Transformer layers
        self.layers = nn.ModuleList([
            InvertedTransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        # Output layer
        self.fc = nn.Linear(d_model * num_patches, output_length)

    def forward(self, x):
        # Patch embedding
        batch_size, num_patches, patch_length, num_features = x.shape
        x = x.view(batch_size, num_patches, -1)  # Flatten features within patches
        x = self.patch_embedding(x)  # (batch_size, num_patches, d_model)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Flatten and predict
        x = x.view(batch_size, -1)  # Flatten patches
        x = self.fc(x)
        return x

# Data Generator
class DataGenerator:
    def __init__(self, data, input_length, output_length, patch_length, num_patches, batch_size, shuffle=True):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.patch_length = patch_length
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data) - input_length - output_length + 1)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs, outputs = [], []
        for i in batch_indices:
            inputs.append(self.data[i:i + self.input_length])
            outputs.append(self.data[i + self.input_length:i + self.input_length + self.output_length, -1])
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        inputs = patchify(inputs, self.patch_length, self.num_patches)
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

data = load_data('/content/drive/MyDrive/msc_data/traffic/traffic.csv')
# Split data into train, validation, and test sets
def split_data(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Split into train and temp (val + test)
    train_data, temp_data = train_test_split(data, train_size=train_ratio, shuffle=False)

    # Split temp into validation and test
    val_data, test_data = train_test_split(temp_data, train_size=val_ratio/(val_ratio + test_ratio), shuffle=False)

    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(data)

# Normalize data
train_data, val_data, test_data, scaler = normalize_data(train_data, val_data, test_data)

# Create data generators
batch_size = 32
train_generator = DataGenerator(train_data, input_length, output_length, patch_length, num_patches, batch_size)
val_generator = DataGenerator(val_data, input_length, output_length, patch_length, num_patches, batch_size)
test_generator = DataGenerator(test_data, input_length, output_length, patch_length, num_patches, batch_size)

# Hyperparameters
d_model = 64           # Model dimension
nhead = 4              # Number of attention heads
num_layers = 2         # Number of transformer layers
dropout = 0.5          # Dropout rate
lr = 1e-4              # Learning rate
epochs = 50            # Number of epochs

# Model, loss, optimizer
num_features = train_data.shape[1]  # Include all features
model = InvertedPatchTST(patch_length, num_patches, d_model, nhead, num_layers, dropout, num_features).to(device)
criterion = nn.MSELoss()  # Add label smoothing if needed
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  # Added weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Learning rate scheduler

# Training loop with early stopping
best_val_loss = float('inf')
patience = 5  # Increased patience
wait = 0
val_loss = 0
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_generator:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
    scheduler.step(val_loss)  # Update learning rate based on validation loss

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_generator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_generator)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# Testing
test_outputs_pred = []
test_targets = []  # To store all targets
model.eval()
for inputs, targets in test_generator:
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Updated mixed precision syntax
            outputs = model(inputs)
        test_outputs_pred.append(outputs.cpu())  # Move predictions to CPU
        test_targets.append(targets.cpu())  # Move targets to CPU

# Concatenate all predictions and targets
test_outputs_pred = torch.cat(test_outputs_pred, dim=0)
test_targets = torch.cat(test_targets, dim=0)

# Compute metrics
mse = mean_squared_error(test_targets.numpy(), test_outputs_pred.numpy())
mae = mean_absolute_error(test_targets.numpy(), test_outputs_pred.numpy())
print(f"Test MSE: {mse}, Test MAE: {mae}")



# Save the trained model and scaler parameters
torch.save(model.state_dict(), 'solar_power_model.pth')
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Save the scaler for the target variable
scaler_target = StandardScaler()
scaler_target.fit(train_data[:, -1].reshape(-1, 1))  # Fit on the target column of the training data
np.save('scaler_target_mean.npy', scaler_target.mean_)
np.save('scaler_target_scale.npy', scaler_target.scale_)

print("Model and scaler parameters saved successfully!")