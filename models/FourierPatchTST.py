__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
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
        
        self.model = PatchTST_backbone(
            c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
            dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
            pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
            subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    def compute_fourier_features(self, x, freq: float, seq_len: int):
        """
        Compute Fourier features for a given frequency.
        Args:
            x: Input tensor [Batch, Input length, Channel]
            freq: Frequency for the Fourier component
            seq_len: Length of the input sequence
        Returns:
            Fourier features [Batch, Input length, 1]
        """
        time = torch.arange(seq_len, device=x.device).float()
        cos_feature = torch.cos(2 * np.pi * freq * time).unsqueeze(0).unsqueeze(-1)
        sin_feature = torch.sin(2 * np.pi * freq * time).unsqueeze(0).unsqueeze(-1)
        return cos_feature, sin_feature


    def forward(self, x):           # x: [Batch, Input length, Channel]
        seq_len = x.shape[1]

        # Compute Fourier features for daily and yearly patterns
        daily_freq = 1 / (24 * 4) # Assuming hourly data
        yearly_freq = 1 / (365 * 24 * 4)  # Assuming hourly data

        daily_cos, daily_sin = self.compute_fourier_features(x, daily_freq, seq_len)
        yearly_cos, yearly_sin = self.compute_fourier_features(x, yearly_freq, seq_len)

        # Concatenate Fourier features with the input
        fourier_features = torch.cat([daily_cos, daily_sin, yearly_cos, yearly_sin], dim=-1)
        fourier_features = fourier_features.repeat(x.size(0), 1, 1)  # Match batch size

        x = torch.cat([x, fourier_features], dim=-1)  # Augmented input

        # Pass through the backbone model
        x = x.permute(0, 2, 1)  # [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # [Batch, Input length, Channel]

        return x