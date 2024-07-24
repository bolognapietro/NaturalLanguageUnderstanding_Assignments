import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LSTM_RNN, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        lstm_out, _  = self.lstm(emb)
        
        output = self.output(lstm_out).permute(0,2,1)
        return output
    
class LSTM_RNN_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LSTM_RNN_DROP, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.emb_dropout = nn.Dropout(emb_dropout)  # Normal dropout
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        self.output = nn.Linear(hidden_size, output_size)
        
        self.out_dropout = nn.Dropout(out_dropout)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(drop1)
        drop2 = self.out_dropout(lstm_out)
        
        output = self.output(drop2).permute(0,2,1)
        return output