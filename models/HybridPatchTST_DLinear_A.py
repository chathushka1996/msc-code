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
        return self.attn(x.mean(dim=1))  # Compute attention weights from mean features

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
        dlinear_output = self.dlinear(x)
        patchtst_output = self.patchtst(x)
        
        # Compute dynamic attention weights
        attn_weights = self.fusion(x)  # [Batch, 2]
        
        # Expand dimensions for broadcasting
        attn_weights = attn_weights.unsqueeze(-1).expand_as(dlinear_output)
        
        # Fuse outputs dynamically
        ensemble_output = attn_weights[:, 0, :] * dlinear_output + attn_weights[:, 1, :] * patchtst_output
        
        return ensemble_output