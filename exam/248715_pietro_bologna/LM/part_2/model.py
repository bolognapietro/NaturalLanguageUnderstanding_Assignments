import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import torch.distributions.bernoulli as brn
    
class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, input, dropout=0.5):
        if not self.training or not dropout:
            return input
        if self.mask is None or self.mask.size() != input.size():
            mask = torch.empty(input.size(), device=input.device).bernoulli_(1 - dropout)
            mask = mask / (1 - dropout)
            self.mask = mask * input

        return self.mask

class LSTM_RNN_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1, weight_tying=False, variational_drop=False):
        super(LSTM_RNN_DROP, self).__init__()
        
        self.weight_tying = weight_tying
        self.variational_drop = variational_drop

        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Add one dropout layer after the embedding layer
        if self.variational_drop:
            self.emb_dropout = VariationalDropout()  # Variational dropout
        else:
            self.emb_dropout = nn.Dropout(emb_dropout)  # Normal dropout           

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.pad_token = pad_index

        # Add one dropout layer after the LSTM layer
        if self.variational_drop:
            self.out_dropout = VariationalDropout()
        else:
            self.out_dropout = nn.Dropout(out_dropout)
            
        # Weight Tying 
        if self.weight_tying:
            self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):

        emb = self.embedding(input_sequence)
        if self.variational_drop:
            emb = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(emb)
        if self.variational_drop:
            lstm_out = self.out_dropout(lstm_out)

        output = self.output(lstm_out).permute(0,2,1)
        return output