import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import DLinear, PatchTST

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # Decomposition module from DLinear
        self.decomp_module = series_decomp(kernel_size=25)
        
        # PatchTST model components
        self.model_trend = PatchTST.Model(configs)
        self.model_res = PatchTST.Model(configs)
    
    def forward(self, x):
        # Decompose the input series
        seasonal_init, trend_init = self.decomp_module(x)
        
        # Permute to match PatchTST input shape
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        # Process each component using PatchTST
        seasonal_output = self.model_res(seasonal_init)
        trend_output = self.model_trend(trend_init)
        
        # Combine the outputs
        x = seasonal_output + trend_output
        
        # Return to standard shape [Batch, Output length, Channel]
        return x.permute(0, 2, 1)
