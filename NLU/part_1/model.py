import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=False, bidirectional=False):
        super(ModelIAS, self).__init__()
               
        # Define the model
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)   

        # If we are using bidirectional LSTM, the hidden size is doubled
        if self.bidirectional:
            hid_size = hid_size * 2 

        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):

        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # Avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            # If we are using bidirectional LSTM, we need to concatenate the hidden states   
            last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        else:
            # Get the last hidden state
            last_hidden = last_hidden[-1,:,:]
        
        # Apply dropout
        if self.dropout:
            utt_encoded = self.dropout(utt_encoded)
            last_hidden = self.dropout(last_hidden)
        
        # Compute slot logits and intent logits
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss

        return slots, intent