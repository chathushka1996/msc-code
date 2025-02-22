from typing import Optional
import torch
import torch.nn as nn
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):  # fusion_type: 'add' or 'concat'
        super(Model, self).__init__()

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
        
        # PatchTST Configuration
        self.patchtst = PatchTST_backbone(
            c_in=c_in, context_window=context_window, target_window=target_window,
            patch_len=patch_len, stride=stride, max_seq_len=1024, 
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
            d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm', attn_dropout=attn_dropout,
            dropout=dropout, act='gelu', key_padding_mask='auto', padding_var=None, 
            attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
            pe='zeros', learn_pe=True, fc_dropout=fc_dropout, head_dropout=head_dropout, 
            padding_patch=padding_patch, pretrain_head=False, head_type='flatten', 
            individual=individual, revin=revin, affine=affine, 
            subtract_last=subtract_last, verbose=False
        )
        
        # DLinear Configuration
        self.decomposition = series_decomp(kernel_size=configs.kernel_size)
        self.linear_seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.linear_trend = nn.Linear(configs.seq_len, configs.pred_len)
        
        # Fusion Mechanism
        self.fusion_type = 'concat'
        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(configs.pred_len * 2, configs.pred_len)
        
    def forward(self, x):
        # PatchTST Prediction
        x_patch = x.permute(0,2,1)  # [Batch, Channel, Seq Len]
        patch_pred = self.patchtst(x_patch)  # [Batch, Channel, Pred Len]
        patch_pred = patch_pred.permute(0,2,1)  # [Batch, Pred Len, Channel]
        
        # DLinear Decomposition
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_pred = self.linear_seasonal(seasonal_init)
        trend_pred = self.linear_trend(trend_init)
        dlinear_pred = seasonal_pred + trend_pred
        dlinear_pred = dlinear_pred.permute(0,2,1)  # [Batch, Pred Len, Channel]
        
        # Fusion
        if self.fusion_type == 'add':
            final_pred = patch_pred + dlinear_pred
        elif self.fusion_type == 'concat':
            fusion_input = torch.cat([patch_pred, dlinear_pred], dim=-1)  # Concatenate along feature dim
            final_pred = self.fusion_layer(fusion_input)
        
        return final_pred
