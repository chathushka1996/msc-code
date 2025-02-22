import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvg(nn.Module):
    """ Moving average block for trend extraction """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size  # Store kernel size explicitly
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # Use self.kernel_size
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """ Decomposes series into trend & seasonal components """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class TransformerBlock(nn.Module):
    """ Transformer Encoder block """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_attn = self.attn(x, x, x)[0]
        x = self.norm1(x + self.dropout(x_attn))
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.decomp = SeriesDecomp(kernel_size=25)
        
        self.conv1 = nn.Conv1d(self.channels, configs.d_model, kernel_size=3, padding=1)
        self.encoder = TransformerBlock(configs.d_model, configs.n_heads, configs.d_ff, dropout=configs.dropout)
        self.fc = nn.Linear(configs.d_model, self.pred_len)  # Ensure correct pred_len
        
    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.conv1(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.conv1(trend.permute(0, 2, 1)).permute(0, 2, 1)
        x = seasonal + trend

        x = self.encoder(x.permute(1, 0, 2))  # [seq_len, batch, d_model]
        x = self.fc(x[-1]).unsqueeze(1)  # [batch, 1, pred_len]
        
        # Debugging print
        print(f"Output shape: {x.shape}, Adjusting to match batch_y if needed")
        
        # Ensure the output shape matches batch_y.shape[-1]
        if x.shape[-1] != self.pred_len:
            x = x[:, :, :self.pred_len]  # Truncate or reshape dynamically

        return x