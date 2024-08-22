from functions import *
from utils import *
from model import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import copy
import math
import numpy as np
import os

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

HID_SIZE = 300  # Hidden size
EMB_SIZE = 300  # Embedding size
N_EPOCHS = 100  # Number of epochs

# Flags
SGD = True
ADAM = False
DROP = False

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

    # Model instantiation
    clip = 5
    vocab_len = len(lang.word2id)

    if DROP:
        model = LSTM_RNN_DROP(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    else:
        model = LSTM_RNN(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)

    # Optimizer
    if SGD and ADAM:
        print("ERROR: select just one optimizer!")
        exit()
    elif ADAM:
        lr = ADAM_LR
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif SGD:
        lr = SGD_LR
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # Initialize weights
    model.apply(init_weights)

    # Loss function
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Training loop
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
        # Train
        ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model, clip)
        array_ppl_train.append(ppl_train)
        array_loss_train.append(loss_train)

        # Evaluate
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss_train).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            array_ppl_dev.append(ppl_dev)
            array_loss_dev.append(loss_dev)

            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            
            # Early stopping
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                # Save the model
                # save_model(model=best_model,filename=f"{model._get_name()}_{optimizer.__class__.__name__}.pt")
                patience = 3
            else:
                patience -= 1

            if patience <= 0:
                break

    best_model.to(DEVICE)

    # Evaluate on the test set
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    
    # Print the result
    print('Test ppl: ', final_ppl)

    array_ppl_dev.append(final_ppl)
    array_ppl_train.append(final_ppl)

    # Plot the results
    # plot_graph(array_ppl_dev, array_ppl_train, array_loss_dev, array_loss_train, 
    #            f"PPL: {model._get_name()} with {optimizer.__class__.__name__}: {SGD_LR if SGD else ADAM_LR} and drop: {DROP} --> {final_ppl}", 
    #            f"LOSS: {model._get_name()} with {optimizer.__class__.__name__}: {SGD_LR if SGD else ADAM_LR} and drop: {DROP} --> {final_ppl}")

if __name__ == "__main__":
    main()