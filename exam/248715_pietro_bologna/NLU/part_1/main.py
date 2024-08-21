from functions import *
from utils import *
from model import ModelIAS

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

import csv
import os

DEVICE = 'cuda:0'

HID_SIZE = 400  # Hidden size
EMB_SIZE = 500  # Embedding size

# Flags
BIDIRECTIONAL = True   # Bidirectional LSTM
DROP = True    # Dropout

PAD_TOKEN = 0   

LR = 0.0001     # Learning rate
clip = 5        # Clip the gradient

# First we get the 10% of the training set, then we compute the percentage of these examples 
portion = 0.10

def main():
    # Load the data
    tmp_train_raw = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "ATIS", "test.json"))

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: 
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    words = sum([x['utterance'].split() for x in train_raw], []) 
                                                            
    corpus = train_raw + dev_raw + test_raw 
    
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # Create the lang object
    lang = Lang(words, intents, slots, cutoff=0)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Model instantiation
    model = ModelIAS(HID_SIZE, out_slot, out_int, EMB_SIZE, vocab_len, pad_index=PAD_TOKEN, dropout=DROP, bidirectional=BIDIRECTIONAL).to(DEVICE)

    # Create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Optimizer and loss function 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    
    # Set the filename for saving the trained model based on the configuration flags
    flags = []
    if BIDIRECTIONAL:
        flags.append('BIDIRECTIONAL')
    if DROP:
        flags.append('DROP')
    
    if flags:
        flags_str = '_'.join(flags)
        filename = f"{model._get_name()}_{flags_str}.pt"
    else:
        filename = f"{model._get_name()}.pt"

    # Training loop
    n_epochs = 200
    patience = 3
    best_f1 = 0
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []

    for x in tqdm(range(1,n_epochs)):
        # Train
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        
        # Evaluate 
        if x % 5 == 0: 
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, _, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            
            # Early stopping
            if f1 > best_f1:
                best_f1 = f1
                # Save the model
                save_model(epoch=x, model=model, optimizer=optimizer, lang=lang, filename=filename)
                patience = 3
            else:
                patience -= 1
            if patience <= 0: 
                break 
    
    # Evaluate on the test set
    results_test, intent_test, loss_dev = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    

    # Print the results
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    # Save the results in a csv file
    # data = {'hid_size': HID_SIZE, 'emb_size': EMB_SIZE, 'n_epochs': n_epochs, 'lr': LR, 'bidir': BIDIRECTIONAL, 'drop': DROP, 'slot F1': results_test['total']['f'], 'accuracy': intent_test['accuracy']}
    # csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    # with open(csv_file, 'a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=data.keys())
    #     writer.writeheader()
    #     writer.writerow(data)

    # Plot the results
    # plot_graph(losses_dev, losses_train,
    #            f"LOSS: bidir {BIDIRECTIONAL} and drop {DROP} with lr {LR}: hid-emb_size {HID_SIZE}-{EMB_SIZE} and epochs {n_epochs} --> slot F1 {results_test['total']['f']} and accuracy {intent_test['accuracy']}",)

if __name__ == "__main__":
    main()