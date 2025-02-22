import torch
import torch.nn as nn

from models import DLinear, PatchTST

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Learnable weights
        self.weight_dlinear = nn.Parameter(torch.tensor(0.5))
        self.weight_patchtst = nn.Parameter(torch.tensor(0.5))
        
        # Initialize both models
        self.dlinear = DLinear(configs)
        self.patchtst = PatchTST(configs)
    
    def forward(self, x):
        # Get predictions from both models
        dlinear_output = self.dlinear(x)
        patchtst_output = self.patchtst(x)
        
        # Apply softmax to ensure weights sum to 1
        weights = torch.softmax(torch.cat([self.weight_dlinear.view(1), self.weight_patchtst.view(1)]), dim=0)
        
        # Weighted sum of outputs
        ensemble_output = (weights[0] * dlinear_output) + (weights[1] * patchtst_output)
        
        return ensemble_output