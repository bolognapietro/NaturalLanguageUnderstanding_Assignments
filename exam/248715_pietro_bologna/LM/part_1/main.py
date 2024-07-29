from functions import *
from utils import *
from model import *

import torch
import torch.optim as optim

import numpy as np
from tqdm import tqdm
import os
import copy
import math
import csv

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

HID_SIZE = 300
EMB_SIZE = 300
N_EPOCHS = 100

# Flags
DROP = True
SGD = True
ADAM = False

# Hyperparameters
SGD_LR = 5
ADAM_LR = 0.001

# Batch sizes
TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

def main():
    # Load the dataset
    train_raw = read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.train.txt"))
    dev_raw = read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.valid.txt"))
    test_raw = read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PennTreeBank", "ptb.test.txt"))

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
    clip = 10
    vocab_len = len(lang.word2id)

    if DROP:
        model = LSTM_RNN_DROP(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    else:
        model = LSTM_RNN(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)

    if ADAM:
        optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR)
    else:
        optimizer = optim.SGD(model.parameters(), lr=SGD_LR)

    model.apply(init_weights)

    # Loss
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Configuration 
    print(f"Configuration: \n\tmodel={model.__class__.__name__}, \n\toptimizer={optimizer.__class__.__name__}, \n\tlr={SGD_LR if SGD else ADAM_LR}, \n\tdrop={DROP}\n")

    # Training
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

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    # Save config and final_ppl to a CSV file
    data = {'model': model.__class__.__name__, 'optimizer': optimizer.__class__.__name__, 'lr': SGD_LR if SGD else ADAM_LR, 'drop': DROP, 'final_ppl': final_ppl}
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)


if __name__ == "__main__":
    main()