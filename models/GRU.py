"""
GRU Model for Time Series Forecasting
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        
        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.pred_len * self.c_out)
        )
        
    def forward(self, x):
        # x: [Batch, Seq_len, Features]
        batch_size = x.size(0)
        
        gru_out, hidden = self.gru(x)
        
        last_hidden = gru_out[:, -1, :]
        
        out = self.fc(last_hidden)
        
        out = out.view(batch_size, self.pred_len, self.c_out)
        
        return out


class BiGRU(nn.Module):
    """Bidirectional GRU for Time Series Forecasting."""
    
    def __init__(self, configs):
        super(BiGRU, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        
        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.pred_len * self.c_out)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        gru_out, hidden = self.gru(x)
        
        last_hidden = gru_out[:, -1, :]
        
        out = self.fc(last_hidden)
        
        out = out.view(batch_size, self.pred_len, self.c_out)
        
        return out
