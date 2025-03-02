import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
    
# Prepare data
train_data = load_data('/content/data/us/train.csv')
val_data = load_data('/content/data/us/val.csv')
test_data = load_data('/content/data/sl/test.csv')

# Normalize data
train_data, val_data, test_data, scaler = normalize_data(train_data, val_data, test_data)

# Create sequences
train_inputs, train_outputs = create_sequences(train_data, input_length, output_length)
val_inputs, val_outputs = create_sequences(val_data, input_length, output_length)
test_inputs, test_outputs = create_sequences(test_data, input_length, output_length)

# Create patches
train_patches = patchify(train_inputs, patch_length, num_patches)
val_patches = patchify(val_inputs, patch_length, num_patches)
test_patches = patchify(test_inputs, patch_length, num_patches)

# Convert to PyTorch tensors and move to GPU
train_patches = torch.tensor(train_patches, dtype=torch.float32).to(device)
train_outputs = torch.tensor(train_outputs, dtype=torch.float32).to(device)
val_patches = torch.tensor(val_patches, dtype=torch.float32).to(device)
val_outputs = torch.tensor(val_outputs, dtype=torch.float32).to(device)
test_patches = torch.tensor(test_patches, dtype=torch.float32).to(device)
test_outputs = torch.tensor(test_outputs, dtype=torch.float32).to(device)


# Hyperparameters
d_model = 64           # Model dimension
nhead = 4              # Number of attention heads
num_layers = 2         # Number of transformer layers
dropout = 0.5          # Dropout rate
lr = 1e-4              # Learning rate
epochs = 50            # Number of epochs
batch_size = 32        # Batch size

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
    for i in range(0, len(train_patches), batch_size):
        inputs = train_patches[i:i + batch_size]
        targets = train_outputs[i:i + batch_size]
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
        for i in range(0, len(val_patches), batch_size):
            inputs = val_patches[i:i + batch_size]
            targets = val_outputs[i:i + batch_size]
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= (len(val_patches) // batch_size)

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
test_batch_size = 16  # Smaller batch size for testing
test_outputs_pred = []
model.eval()
for i in range(0, len(test_patches), test_batch_size):
    batch = test_patches[i:i + test_batch_size]
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Enable mixed precision
            outputs = model(batch)
        test_outputs_pred.append(outputs.cpu())  # Move predictions to CPU
test_outputs_pred = torch.cat(test_outputs_pred, dim=0)

# Compute metrics
mse = mean_squared_error(test_outputs.cpu().numpy(), test_outputs_pred.numpy())
mae = mean_absolute_error(test_outputs.cpu().numpy(), test_outputs_pred.numpy())
print(f"Test MSE: {mse}, Test MAE: {mae}")