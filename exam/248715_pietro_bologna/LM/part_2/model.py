import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = p

    def forward(self, input):
        if not self.training or not self.dropout == 0:
            return input
        
        mask = input.new_empty(input.size(0), 1, input.size(2)).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)
        mask = mask.expand_as(input)

        return mask * input

class LSTM_RNN_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1, weight_tying=False, variational_drop=False):
        super(LSTM_RNN_DROP, self).__init__()

        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Add one dropout layer after the embedding layer
        if variational_drop:
            self.emb_dropout = VariationalDropout(p=emb_dropout)  # Variational dropout
        else:
            self.emb_dropout = nn.Dropout(emb_dropout)  # Normal dropout           

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Add one dropout layer after the LSTM layer
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
        emb = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)

        output = self.output(lstm_out).permute(0,2,1)
        return output