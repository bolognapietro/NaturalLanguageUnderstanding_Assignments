import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
# Variational Dropout
class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = p

    def forward(self, input):
        # Dropout is only applied during training
        if not self.training or not self.dropout:
            return input
        
        mask = input.new_empty(input.size(0), 1, input.size(2)).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)
        mask = mask.expand_as(input)

        return mask * input

# LSTM model
class LSTM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1, weight_tying=False, variational_drop=False):
        super(LSTM_RNN, self).__init__()

        # Define the model
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Dropout layer after the embedding layer
        if variational_drop:
            self.emb_dropout = VariationalDropout(p=emb_dropout)  # Variational dropout
        else:
            self.emb_dropout = nn.Dropout(emb_dropout)  # Normal dropout           

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Dropout layer before the last linear layer
        if variational_drop:
            self.out_dropout = VariationalDropout(p=out_dropout)
        else:
            self.out_dropout = nn.Dropout(out_dropout)
        
        self.output = nn.Linear(hidden_size, output_size)

        # Weight Tying 
        if weight_tying:
            self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # Apply dropout to the embedding layer's output
        drop1 = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(drop1)
        # Apply dropout to the LSTM layer's output
        drop2 = self.out_dropout(lstm_out)

        output = self.output(drop2).permute(0,2,1)
        
        return output