__all__ = ['PatchTST']

# Cell
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import statsmodels.api as sm
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        self.seasonal_period = 96
        self.kernel_size = 97
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        self.patch_model = PatchTST_backbone(
            c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
            dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
            pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
            subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    def decompose_single_series_gpu(self, series):
        """Performs approximate STL decomposition on a single time series using GPU."""
        # series shape: [batch_size, seq_len]
        batch_size, seq_len = series.shape

        # Low-pass filter for trend extraction
        trend_filter = torch.ones(1, 1, self.kernel_size, device=series.device) / self.kernel_size
        padding = self.kernel_size // 2  # Ensure output size matches input size
        trend = F.conv1d(series.unsqueeze(1), trend_filter, padding=padding)
        trend = trend.squeeze(1)

        # Ensure trend matches series size (trim excess, if any)
        trend = trend[:, :seq_len]

        # Seasonal extraction: subtract trend and apply periodic smoothing
        detrended = series - trend
        seasonal_filter = torch.zeros(seq_len, device=series.device)
        seasonal_filter[::self.seasonal_period] = 1.0
        seasonal_filter = seasonal_filter / seasonal_filter.sum()  # Normalize

        # Convolution with periodic filter
        seasonal = F.conv1d(
            detrended.unsqueeze(1),
            seasonal_filter.view(1, 1, -1),
            padding=self.seasonal_period // 2,
        )
        seasonal = seasonal.squeeze(1)

        # Ensure seasonal matches series size (trim excess, if any)
        seasonal = seasonal[:, :seq_len]

        # Residual computation
        residual = series - trend - seasonal

        return trend, seasonal, residual

    def forward(self, x):
        """Forward pass for the model."""
        batch_size, seq_len, _ = x.shape

        # Decompose batch in parallel on GPU
        x_series = x[:, :, 0]  # Shape: [batch_size, seq_len]
        trend, seasonal, residual = self.decompose_single_series_gpu(x_series)

        # Predict residuals using PatchTST
        residual_pred = self.patch_model(residual.unsqueeze(-1).permute(0, 2, 1)).permute(0, 2, 1)

        # Combine trend, seasonal, and residual predictions
        output = trend.unsqueeze(-1) + seasonal.unsqueeze(-1) + residual_pred

        return output