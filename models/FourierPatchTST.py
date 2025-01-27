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
        self.statistical_projection = nn.Linear(2, configs.d_model)  # Assuming 2 features (daily & yearly)
        self.final_projection = nn.Linear(configs.d_model * 2, configs.d_model)  # Combine statistical and learned features
        
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
    
    def extract_seasonal_features(self, data, freq_daily=96, freq_yearly=35040):
        """
        Extract daily and yearly features using sinusoidal models.
        Args:
            data (numpy array): Time series data of shape [T,]
            freq_daily (int): Number of steps corresponding to daily frequency.
            freq_yearly (int): Number of steps corresponding to yearly frequency.
        Returns:
            numpy array: Array of shape [T, 2] with daily and yearly patterns.
        """
        t = np.arange(len(data))
        daily_pattern = np.sin(2 * np.pi * t / freq_daily)
        yearly_pattern = np.sin(2 * np.pi * t / freq_yearly)
        return np.column_stack((daily_pattern, yearly_pattern))


    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # [Batch, Channel, Input length]
        learned_features = self.model(x)  # [Batch, Channels, Input length]
        stats_features = self.statistical_projection(stats_features)  # [Batch, Input length, D_model]
        combined = torch.cat((learned_features.permute(0, 2, 1), stats_features), dim=-1)
        output = self.final_projection(combined).permute(0, 2, 1)  # [Batch, Channel, Input length]
        return output