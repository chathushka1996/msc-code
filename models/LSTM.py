"""
LSTM Model for Time Series Forecasting
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
        
        self.lstm = nn.LSTM(
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
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        last_hidden = lstm_out[:, -1, :]
        
        out = self.fc(last_hidden)
        
        out = out.view(batch_size, self.pred_len, self.c_out)
        
        return out


class LSTMSeq2Seq(nn.Module):
    """Sequence-to-Sequence LSTM with attention-like output generation."""
    
    def __init__(self, configs):
        super(LSTMSeq2Seq, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        
        self.encoder = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.decoder = nn.LSTM(
            input_size=self.c_out,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(self.hidden_size, self.c_out)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        _, (hidden, cell) = self.encoder(x)
        
        decoder_input = x[:, -1:, :self.c_out]
        
        outputs = []
        for _ in range(self.pred_len):
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.fc(decoder_out)
            outputs.append(out)
            decoder_input = out
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
