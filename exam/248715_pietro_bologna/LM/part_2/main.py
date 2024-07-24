from functions import *
from utils import *
from model import LSTM_RNN_DROP

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import copy
import math
import numpy as np
import os
from functools import partial

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

HID_SIZE = 600
EMB_SIZE = 600
N_EPOCHS = 100
NON_MONO = 3

# Flags
SGD = True
ADAM = False
ASGD = True
WEIGH_TYING = True
VARIATIONA_DROP = True

# Hyperparameters
SGD_LR = 5
ADAM_LR = 0.001

# Batch sizes
TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

def main():
    # Load the dataset
    train_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.train.txt")
    dev_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.valid.txt")
    test_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.test.txt")

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
    model = LSTM_RNN_DROP(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"], weight_tying=WEIGH_TYING, variational_drop=VARIATIONA_DROP).to(device)
    model.apply(init_weights)

    if ADAM:
        lr = ADAM_LR
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        lr = SGD_LR
        optimizer = optim.SGD(model.parameters(), lr=lr)

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

            # If the model is using ASGD, we need to update the learning rate
            if 't0' in optimizer.param_groups[0] and ASGD:
                tmp = {}
                for param in model.parameters():
                    tmp[param] = param.data.clone()
                    param.data = optimizer.state[param]['ax'].clone()

                ppl_dev, loss_dev2 = eval_loop(dev_loader, criterion_eval, model)

                print("[AvSGD]")

                for param in model.parameters():
                    param.data = tmp[param].clone()
            
            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

                string = "[SDG]"

                if ASGD and SGD and 't0' not in optimizer.param_groups[0] and (len(array_loss_dev) > NON_MONO and loss_dev > min(array_loss_dev[:-NON_MONO])):
                    
                    # I switch to ASGD if the development loss has not improved over a defined number of epochs and if the loss of the current epoch is
                    # higher than the minimum loss from a specific number of previous epochs

                    string = "Switching to [ASGD]"
                    optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.)

            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            array_ppl_dev.append(ppl_dev)
            array_loss_dev.append(loss_dev)
            losses_dev.append(np.asarray(loss_dev).mean())

            pbar.set_description(string + " -> PPL: %f" % ppl_dev)
            
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    
    array_loss_dev.append(final_ppl)
    array_ppl_dev.append(final_ppl)

    print('Test ppl: ', final_ppl)

if __name__ == "__main__":
    main()