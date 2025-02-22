import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
import numpy as np
from statsmodels.tsa.seasonal import STL


def series_decomposition(x, seasonal=24):
    """
    Decomposes the input time series into trend, seasonal, and remainder components.
    Applies decomposition separately to each feature.
    """
    B, L, D = x.shape  # Batch, Sequence Length, Features
    trend, seasonal_comp, remainder = [], [], []

    # Ensure seasonal period is valid
    if seasonal < 2:
        seasonal = max(2, L // 10)  # Fallback to a reasonable value

    for i in range(D):  # Iterate over each feature
        feature_series = x[:, :, i]  # Extract single feature across batch
        decomposed = [STL(feature_series[j].cpu().numpy(), period=seasonal).fit() for j in range(B)]

        trend.append(torch.tensor([d.trend for d in decomposed]).to(x.device))
        seasonal_comp.append(torch.tensor([d.seasonal for d in decomposed]).to(x.device))
        remainder.append(torch.tensor([d.resid for d in decomposed]).to(x.device))

    return torch.stack(trend, dim=-1), torch.stack(seasonal_comp, dim=-1), torch.stack(remainder, dim=-1)


def create_patches(x, patch_size=16):
    """
    Splits the time series into overlapping patches.
    """
    B, L, D = x.shape
    num_patches = L // patch_size
    x = x[:, :num_patches * patch_size, :].reshape(B, num_patches, patch_size, D)
    return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.patch_size = configs.patch_size if hasattr(configs, 'patch_size') else 16
        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Encoder with patches
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Series Decomposition
        trend, seasonal_comp, remainder = series_decomposition(x_enc)
        x_enc = torch.cat([trend, seasonal_comp, remainder], dim=-1)
        
        # Patching
        x_enc = create_patches(x_enc, self.patch_size)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        return (dec_out[:, -self.pred_len:, :], attns) if self.output_attention else dec_out[:, -self.pred_len:, :]