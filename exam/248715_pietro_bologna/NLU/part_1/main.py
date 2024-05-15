# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import spacy
import numpy as np
from scipy.spatial.distance import cosine
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

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

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Vocab is computed only on training set 
    # We add two special tokens end of sentence and padding 
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    len(vocab)

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    #* ---------- Setup ----------
    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    hid_size = 200
    emb_size = 200

    # Don't forget to experiment with a lower training batch size
    # Increasing the back propagation steps can be seen as a regularization step
    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 2 # This is definitely not good for SGD
    clip = 5 # Clip the gradient
    device = 'cuda:0'

    vocab_len = len(lang.word2id)

    model = LSTM_RNN_DROP(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    #! Replace SGD with AdamW (1.3)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    #* ---------- Run ----------
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

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)