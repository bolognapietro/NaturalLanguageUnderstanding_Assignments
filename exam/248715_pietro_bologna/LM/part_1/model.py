import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LSTM model
class LSTM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LSTM_RNN, self).__init__()

        # Define the model
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        lstm_out, _  = self.lstm(emb)
        
        output = self.output(lstm_out).permute(0,2,1)
        return output
    
# LSTM model with dropout
class LSTM_RNN_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LSTM_RNN_DROP, self).__init__()
        
        # Define the model
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Dropout layer after the embedding layer
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        # Dropout layer before the last linear layer
        self.out_dropout = nn.Dropout(out_dropout)

        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # Apply dropout to the embedding layer's output
        drop1 = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(drop1)
        # Apply dropout to the LSTM layer's output
        drop2 = self.out_dropout(lstm_out)
        
        output = self.output(drop2).permute(0,2,1)
        
        return output