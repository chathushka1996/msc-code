from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Define model architecture (same as during training)
class InvertedPatchTST(nn.Module):
    def __init__(self, patch_length, output_length, num_patches, d_model, nhead, num_layers, dropout, num_features):
        super(InvertedPatchTST, self).__init__()
        self.patch_length = patch_length
        self.num_patches = num_patches
        self.d_model = d_model
        self.patch_embedding = nn.Linear(patch_length * num_features, d_model)
        self.layers = nn.ModuleList([
            InvertedTransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model * num_patches, output_length)

    def forward(self, x):
        batch_size, num_patches, patch_length, num_features = x.shape
        x = x.reshape(batch_size, num_patches, -1)  # Use reshape instead of view
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

# Hyperparameters (same as during training)
input_length = 96      # Input sequence length
patch_length = 12      # Length of each patch
num_patches = input_length // patch_length  # Number of patches
d_model = 64           # Model dimension
nhead = 4              # Number of attention heads
num_layers = 2         # Number of transformer layers
dropout = 0.5          # Dropout rate
num_features = 10      # Number of features in the dataset (excluding 'date' and 'Solar Power Output')

def select_model_scalers(prediction_steps):
    # Load the scalers
    scaler_input = StandardScaler()
    scaler_input.mean_ = np.load(f'save/{prediction_steps}/scaler_mean.npy')
    scaler_input.scale_ = np.load(f'save/{prediction_steps}/scaler_scale.npy')

    scaler_target = StandardScaler()
    scaler_target.mean_ = np.load(f'save/{prediction_steps}/scaler_target_mean.npy')
    scaler_target.scale_ = np.load(f'save/{prediction_steps}/scaler_target_scale.npy')

    output_length = prediction_steps    # Output sequence length
    # Load the trained model
    # Ensure the model is loaded onto the CPU
    model = InvertedPatchTST(patch_length, output_length, num_patches, d_model, nhead, num_layers, dropout, num_features)

    # Load the model with map_location set to 'cpu'
    model.load_state_dict(torch.load(f'save/{prediction_steps}/solar_power_model.pth', map_location=torch.device('cpu')))
    model.eval()

    return model, scaler_input, scaler_target

# Load the dataset
csv_file_path = 'data/sl_t/test.csv'
data = pd.read_csv(csv_file_path)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# Function to get the previous 96 time steps
def get_previous_96_steps(target_date):
    target_date = pd.to_datetime(target_date)
    start_date = target_date - timedelta(minutes=15 * 95)  # 96 steps of 15 minutes each
    subset = data[(data['date'] >= start_date) & (data['date'] <= target_date)]
    if len(subset) != 96:
        raise ValueError(f"Expected 96 time steps, but got {len(subset)}")
    # return subset.drop(columns=['date', 'Solar Power Output']).values  # Drop date and target column
    return subset.drop(columns=['date']).values  # Drop date and target column

# Function to get the previous 96 time steps
def get_next_steps(target_date, prediction_steps):
    target_date = pd.to_datetime(target_date)
    print(target_date)
    end_date = target_date + timedelta(minutes=15 * prediction_steps)  # 96 steps of 15 minutes each
    subset = data[(data['date'] <= end_date) & (data['date'] > target_date)]
    if len(subset) != prediction_steps:
        raise ValueError(f"Expected {prediction_steps} time steps, but got {len(subset)}")
    # return subset.drop(columns=['date', 'Solar Power Output']).values  # Drop date and target column
    
    result = []

    # Iterate over the subset and create the value dictionary for each row
    for _, row in subset.iterrows():
        value = {
            "datetime": str(row['date']),
            "solar_power_real": float(row['Solar Power Output'])
        }
        result.append(value)
    
    return result

# Function to patchify the input data
def patchify(data, patch_length, num_patches):
    patches = []
    for i in range(num_patches):
        start = i * patch_length
        end = start + patch_length
        patches.append(data[:, start:end, :])
    return np.stack(patches, axis=1)

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date and time from the user
        user_input = request.json['date_time']
        prediction_steps = request.json['prediction_steps']
        target_date = pd.to_datetime(user_input)

        model, scaler_input, scaler_target = select_model_scalers(prediction_steps)
        # Get the previous 96 time steps
        previous_96_steps = get_previous_96_steps(target_date)

        # Normalize the input data
        previous_96_steps_scaled = scaler_input.transform(previous_96_steps)

        # Patchify the input data
        previous_96_steps_patches = patchify(previous_96_steps_scaled[np.newaxis, :, :], patch_length, num_patches)

        # Convert to tensor and move to device
        previous_96_steps_patches = torch.tensor(previous_96_steps_patches, dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            predicted_output_scaled = model(previous_96_steps_patches)  # Shape: (1, 720)

        # Move predictions to CPU and convert to numpy
        predicted_output_scaled = predicted_output_scaled.cpu().numpy().flatten()  # Shape: (720,)

        # Reverse scaling for the predictions
        predicted_output_unscaled = scaler_target.inverse_transform(predicted_output_scaled.reshape(-1, 1)).flatten()  # Shape: (720,)
        print(target_date)
        value_array = []
        target_date_temp = target_date
        for i in predicted_output_unscaled:
            pred_date = target_date_temp + timedelta(minutes=15)
            target_date_temp = pred_date
            value = {
                "datetime": str(pred_date),
                "solar_power_pred": float(i)
            }
            value_array.append(value)

        # # Return the predicted output
        # return jsonify({
        #     'date_time': user_input,
        #     'predicted_solar_power_output': predicted_output_unscaled.tolist()
        # })
        result = get_next_steps(target_date, prediction_steps)
        return {
            "prediction": value_array,
            "real": result
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)