import torch
import torch.nn as nn

from models import DLinear, PatchTST

class Model(nn.Module):
    def __init__(self, configs, weight_dlinear=0.5, weight_patchtst=0.5):
        super(Model, self).__init__()
        self.weight_dlinear = weight_dlinear
        self.weight_patchtst = weight_patchtst
        
        # Initialize both models
        self.dlinear = DLinear.Model(configs)
        self.patchtst = PatchTST.Model(configs)
    
    def forward(self, x):
        # Get predictions from both models
        dlinear_output = self.dlinear(x)
        patchtst_output = self.patchtst(x)
        
        # Weighted sum of outputs
        ensemble_output = (self.weight_dlinear * dlinear_output) + (self.weight_patchtst * patchtst_output)
        
        return ensemble_output