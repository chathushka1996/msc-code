import torch
import torch.nn as nn

from models import DLinear, PatchTST

class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Two models
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Instead of taking the mean over time and features, pass the full input
        return self.attn(x.mean(dim=1))  # Mean over time axis (dim=1)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Initialize models
        self.dlinear = DLinear.Model(configs)
        self.patchtst = PatchTST.Model(configs)
        
        # Attention mechanism for adaptive fusion
        self.fusion = AttentionFusion(configs.enc_in)
    
    def forward(self, x):
        # Get model outputs
        dlinear_output = self.dlinear(x)  # Shape: [Batch, Output_length, Channels]
        patchtst_output = self.patchtst(x)  # Shape: [Batch, Output_length, Channels]
        
        # Compute dynamic attention weights
        attn_weights = self.fusion(x)  # [Batch, 2]
        
        # Expand dimensions for broadcasting
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)  # Shape: [Batch, 2, 1, 1]
        attn_weights = attn_weights.expand(-1, -1, dlinear_output.shape[1], dlinear_output.shape[2])  # [Batch, 2, Output_length, Channels]
        
        # Fuse outputs dynamically
        ensemble_output = attn_weights[:, 0] * dlinear_output + attn_weights[:, 1] * patchtst_output
        
        return ensemble_output