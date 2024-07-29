# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *

import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.optim as optim
from tqdm import tqdm

DEVICE = 'cuda:0'

HID_SIZE = 768
EMB_SIZE = 300

LR = 0.0001     # Learning rate
clip = 5        # Clip the gradient

# First we get the 10% of the training set, then we compute the percentage of these examples 
portion = 0.10

def main():
    tmp_train_raw = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "ATIS", "test.json"))
    
    
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
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

    lang = Lang(words, intents, slots, cutoff=0)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = JointBERT(HID_SIZE, out_slot, out_int).to(DEVICE)
    model.apply(init_weights)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.slot2id['pad'])
    criterion_intents = nn.CrossEntropyLoss() 
    
    #! SINGLE RUN
    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)

        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']

            if f1 > best_f1:
                best_f1 = f1
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break 
            
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    

    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

if __name__ == "__main__":
    main()
