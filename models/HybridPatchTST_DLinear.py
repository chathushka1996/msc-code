import torch
import torch.nn as nn
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class Model(nn.Module):
    def __init__(self, configs, fusion_type='add'):  # fusion_type: 'add' or 'concat'
        super(Model, self).__init__()
        
        # PatchTST Configuration
        self.patchtst = PatchTST_backbone(
            c_in=configs.enc_in, context_window=configs.seq_len, target_window=configs.pred_len,
            patch_len=configs.patch_len, stride=configs.stride, max_seq_len=1024, 
            n_layers=configs.e_layers, d_model=configs.d_model, n_heads=configs.n_heads, 
            d_k=None, d_v=None, d_ff=configs.d_ff, norm='BatchNorm', attn_dropout=configs.attn_dropout,
            dropout=configs.dropout, act='gelu', key_padding_mask='auto', padding_var=None, 
            attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
            pe='zeros', learn_pe=True, fc_dropout=configs.fc_dropout, head_dropout=configs.head_dropout, 
            padding_patch=configs.padding_patch, pretrain_head=False, head_type='flatten', 
            individual=configs.individual, revin=configs.revin, affine=configs.affine, 
            subtract_last=configs.subtract_last, verbose=False
        )
        
        # DLinear Configuration
        self.decomposition = series_decomp(kernel_size=configs.kernel_size)
        self.linear_seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.linear_trend = nn.Linear(configs.seq_len, configs.pred_len)
        
        # Fusion Mechanism
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
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
