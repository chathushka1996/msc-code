import torch
import torch.nn as nn
from models import Informer

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)

    def forward(self, x):
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # Moving average for trend
        seasonal_residual = x - trend  # Residual = original - trend
        return trend, seasonal_residual

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        print(f"output_attention: {configs.output_attention}")
        self.decomposition = SeriesDecomposition(kernel_size=configs.moving_avg)
        
        # Informer models for Trend, Seasonal, Residual
        self.trend_model = Informer.Model(configs)
        self.seasonal_model = Informer.Model(configs)
        self.residual_model = Informer.Model(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        trend, seasonal_residual = self.decomposition(x_enc)
        seasonal, residual = self.decomposition(seasonal_residual)

        trend_pred = self.trend_model(trend, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        seasonal_pred = self.seasonal_model(seasonal, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        residual_pred = self.residual_model(residual, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)

        return trend_pred + seasonal_pred + residual_pred