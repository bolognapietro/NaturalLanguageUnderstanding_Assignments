from functions import *
from utils import *
from model import *
import spacy
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

HID_SIZE = 200
EMB_SIZE = 200
N_EPOCHS = 100

# Flags
DROP = True
SGD = True
ADAM = False

# Hyperparameters
SGD_LR = 1.5
ADAM_LR = 0.001

# Batch sizes
TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

def main():
    # Load the dataset
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Create the vocabulary
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    # Create the language
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Create the datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Model
    clip = 5
    vocab_len = len(lang.word2id)

    if DROP:
        model = LSTM_RNN_DROP(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    else:
        model = LSTM_RNN(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)

    if ADAM:
        optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)
    else:
        optimizer = optim.SGD(model.parameters(), lr=SGD_LR)

    model.apply(init_weights)

    # Loss
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, N_EPOCHS))
    array_ppl_train = []
    array_loss_train = []
    array_ppl_dev = []
    array_loss_dev = []

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

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

if __name__ == "__main__":
    main()