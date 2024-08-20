import json
from pprint import pprint
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functions import *
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

DEVICE = 'cuda:0'

PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, intents, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        vocab['unk'] = len(vocab)
        return vocab
    
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        self.attention_masks = []
        self.slot_ids = []
        self.utt_ids = []
        self.intent_ids = []
        self.token_type_ids = []
        
        for x in dataset:
            self.utterances.append("[CLS] " + x['utterance'] + " [SEP]")
            self.slots.append("O " + x['slots'] + " O")
            self.intents.append(x['intent'])

        self.utt_ids, self.slot_ids, self.attention_mask, self.token_type_ids = self.mapping_seq(self.utterances, self.slots, tokenizer, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        intent = self.intent_ids[idx]
        slots = torch.Tensor(self.slot_ids[idx])
        att = torch.Tensor(self.attention_mask[idx])
        token_type = torch.Tensor(self.token_type_ids[idx])
        sample = {'utterance': utt, 'intent': intent, 'slots': slots, 'attention_mask': att, 'token_type_ids': token_type}
        return sample
    
    # Auxiliary methods
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data_utt, data_slot, tokenizer, mapper): # Map sequences to number 
        res_utt = []
        res_slot = []
        res_att = []
        res_tok_type_id = []

        for seq, slot in zip(data_utt, data_slot):
            tmp_seq = []
            tmp_slot = []
            tmp_att = []
            tmp_tok_type_id = []
            
            for word, elem in zip(seq.split(), slot.split(' ')):
                tmp_att.append(1)
                tmp_tok_type_id.append(0)
                word_tokens = tokenizer(word) 
                word_tokens = word_tokens[1:-1]

                tmp_seq.extend(word_tokens['input_ids'])
                tmp_slot.extend([mapper[elem]] + [mapper['pad']]*(len(word_tokens['input_ids'])-1))             

                for _ in range(len(word_tokens['input_ids'])-1):
                    tmp_att.append(0)
                    tmp_tok_type_id.append(0)  

            res_utt.append(tmp_seq)
            res_slot.append(tmp_slot)
            res_att.append(tmp_att)
            res_tok_type_id.append(tmp_tok_type_id)

        return res_utt, res_slot, res_att, res_tok_type_id

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    attention, _ = merge(new_item['attention_mask'])
    token_type_ids, _ = merge(new_item['token_type_ids'])

    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected DEVICE
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    attention = attention.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["attention_mask"] = attention
    new_item["token_type_ids"] = token_type_ids  
    new_item["slots_len"] = y_lengths
    return new_item