__all__ = ['PatchTST']

# Cell
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
        
        def decompose(self, series, seasonal):
            """
            Perform STL decomposition.
            Args:
                series: Time series data (numpy array).
                seasonal: Seasonal period for decomposition.
            Returns:
                trend, seasonal, residual: Decomposed components.
            """
            stl = sm.tsa.STL(series, seasonal=seasonal).fit()
            return stl.trend, stl.seasonal, stl.resid
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        # Decompose input series for yearly pattern
        x_np = x.cpu().numpy()
        trend_yearly, seasonal_yearly, residual_yearly = [], [], []
        for i in range(x_np.shape[0]):
            trend, seasonal, resid = self.decompose(x_np[i, :, 0], seasonal=35040)  # Yearly seasonality
            trend_yearly.append(trend)
            seasonal_yearly.append(seasonal)
            residual_yearly.append(resid)

        # Convert yearly components back to tensors
        trend_yearly = torch.tensor(trend_yearly, device=x.device, dtype=x.dtype).unsqueeze(-1)
        seasonal_yearly = torch.tensor(seasonal_yearly, device=x.device, dtype=x.dtype).unsqueeze(-1)
        residual_yearly = torch.tensor(residual_yearly, device=x.device, dtype=x.dtype).unsqueeze(-1)

        # Decompose residual for daily pattern
        trend_daily, seasonal_daily, residual_daily = [], [], []
        residual_np = residual_yearly.squeeze(-1).cpu().numpy()
        for i in range(residual_np.shape[0]):
            trend, seasonal, resid = self.decompose(residual_np[i], seasonal=96)  # Daily seasonality
            trend_daily.append(trend)
            seasonal_daily.append(seasonal)
            residual_daily.append(resid)

        # Convert daily components back to tensors
        trend_daily = torch.tensor(trend_daily, device=x.device, dtype=x.dtype).unsqueeze(-1)
        seasonal_daily = torch.tensor(seasonal_daily, device=x.device, dtype=x.dtype).unsqueeze(-1)
        residual_daily = torch.tensor(residual_daily, device=x.device, dtype=x.dtype).unsqueeze(-1)

        # Combine components
        combined_trend = trend_yearly + trend_daily
        combined_seasonal = seasonal_yearly + seasonal_daily

        # Predict residual using PatchTST
        residual_pred = self.patch_model(residual_daily.permute(0, 2, 1)).permute(0, 2, 1)

        # Final output: trend + seasonal + residual prediction
        output = combined_trend[:, -self.pred_len:, :] + combined_seasonal[:, -self.pred_len:, :] + residual_pred
        return output