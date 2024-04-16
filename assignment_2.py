                                            # Neural Language Understanding
                                            # Assignment: 1
                                            # Part: 2

# Run this if you are on Colab
# !python -m spacy download en_core_web_lg
# !wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt
# !wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt
# !wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt


# # Mandatory Exam Exercise
# ## Part 1 (4 points)
# In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment.  For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
# <br>
# One of the important tasks of training a neural network is  hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular <b>the learning rate</b>). 
# These are two links to the state-of-the-art papers which use vanilla RNN [paper1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5947611), [paper2](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf). 
# 
# **Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***).
# 
# 1. Replace RNN with a Long-Short Term Memory (LSTM) network --> [link](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
# 2. Add two dropout layers: --> [link](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
#     - one after the embedding layer, 
#     - one before the last linear layer
# 3. Replace SGD with AdamW --> [link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)


import spacy
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os

nlp = spacy.load('en_core_web_lg')
txt = 'metropolis'
doc = nlp(txt)

tok = doc[0]  # let's take Rome

print("string:", tok.text)
print("vector dimension:", len(tok.vector))
print("spacy vector norm:", tok.vector_norm)

# let's get Paris & compare its vector to rome
paris = nlp('city')[0]
print(paris.text)

print("spacy CosSim({}, {}):".format(tok.text, paris.text), tok.similarity(paris))
print("scipy CosSim({}, {}):".format(tok.text, paris.text), 1 - cosine(tok.vector, paris.vector))

tok2 = nlp('computer')[0]
print(tok2.text)
print("spacy CosSim({}, {}):".format(tok.text, tok2.text), tok.similarity(tok2))
print("scipy CosSim({}, {}):".format(tok.text, tok2.text), 1 - cosine(tok.vector, tok2.vector))

# RNN Elman version
# We are not going to use this since for efficiently purposes it's better to use the RNN layer provided by pytorch  

class RNN_cell(nn.Module):
    def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()
        
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)
        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output 

class LSTM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LSTM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output

class LSTM_RNN_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LSTM_RNN_DROP, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Add one dropout layer after the embedding layer
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Add one dropout layer before the last linear layer
        self.out_dropout = nn.Dropout(out_dropout)

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.out_dropout(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output
    
DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu

# Loading the corpus 
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

# Vocab is computed only on training set 
# We add two special tokens end of sentence and padding 
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
len(vocab)

# This class computes and stores our vocab 
# Word to ids and ids to word
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    
lang = Lang(train_raw, ["<pad>", "<eos>"])

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    
    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

from functools import partial
from torch.utils.data import DataLoader

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

# Dataloader instantiation
# You can reduce the batch_size if the GPU memory is not enough
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return    

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

# Experiment also with a smaller or bigger model by changing hid and emb sizes
# A large model tends to overfit
hid_size = 200
emb_size = 300

# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step

# With SGD try with an higher learning rate (> 1 for instance)
lr = 0.0015 # This is definitely not good for SGD
clip = 5 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LSTM_RNN_DROP(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
model.apply(init_weights)

optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')












n_epochs = 100
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1,n_epochs))
array_ppl_train = []
array_loss_train = []
array_ppl_dev = []
array_loss_dev = []

#If the PPL is too high try to change the learning rate
for epoch in pbar:
    ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
    array_ppl_train.append(ppl_train)
    array_loss_train.append(loss_train)

    if epoch % 1 == 0:
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss_train).mean())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        array_ppl_dev.append(ppl_dev)
        array_loss_dev.append(loss_dev)

        losses_dev.append(np.asarray(loss_dev).mean())
        pbar.set_description("PPL: %f" % ppl_dev)
        
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1
            # lr /= 2

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print('Test ppl: ', final_ppl)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sampled_epochs, array_ppl_train, marker='o', label='ppl train')  # Add marker='o' for points
ax.plot(sampled_epochs, array_ppl_dev, marker='o', label='ppl dev')  # Add marker='o' for points

ax.set_xlabel('Epochs')  # Set x-axis label
ax.set_ylabel('Perplexity')  # Set y-axis label

ax.legend()

path_folder = "/home/disi/NLU-2024-Labs/assignment_1/part_2/A_1"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

plt.savefig(f"{path_folder}/ppl_plot.pdf")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sampled_epochs, array_loss_train, marker='o', label='loss train')  # Add marker='o' for points
ax.plot(sampled_epochs, array_loss_dev, marker='o', label='loss dev')  # Add marker='o' for points

ax.set_xlabel('Epochs')  # Set x-axis label
ax.set_ylabel('Loss')  # Set y-axis label

ax.legend()

plt.savefig(f"{path_folder}/loss_plot.pdf")

# Text to be saved
text_to_save = f"Parameters:\n\nepochs = {epoch-1}\nhid_layers = {hid_size}\nemb_layers = {emb_size}\nclip = {clip} \n\nBest ppl = {final_ppl}"

# Specify the file name
file_name = f"{path_folder}/parameters.txt"

# Open the file in write mode ("w" flag)
with open(file_name, "w") as file:
    # Write the text to the file
    file.write(text_to_save)
