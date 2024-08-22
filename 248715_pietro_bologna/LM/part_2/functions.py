import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Function used to train the model on a training dataset
def train_loop(data, optimizer, criterion, model, clip=5):

    # Set the model in training mode
    model.train()
    loss_array = []
    number_of_tokens = []
    
    # Loop over the dataset
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        
        # Inference
        output = model(sample['source'])
        
        # Compute the loss
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        
        # Backward pass
        loss.backward()

        # Clip the gradient to avoid explosioning gradients 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        
        # Update the weights
        optimizer.step()
        
    # Compute the perplexity and the loss
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    train_loss = sum(loss_array) / sum(number_of_tokens)

    return ppl, train_loss    

# Function used to evaluate the model on a validation/test dataset
def eval_loop(data, eval_criterion, model):

    # Set the model in evaluation mode
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad(): # It used to avoid the creation of computational graph
        
        # Loop over the dataset
        for sample in data:
            # Inference
            output = model(sample['source'])
            
            # Compute the loss
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    # Compute the perplexity and the loss
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    eval_loss = sum(loss_array) / sum(number_of_tokens)

    return ppl, eval_loss

# Function used to initialize the weights of the model
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

# Function used to plot the loss values and the perplexity values
def plot_graph(ppl_dev, ppl_train, losses_dev, losses_train, filename, filename1):

    y1 = ppl_dev[:-1]  # Last value is best_ppl 
    y2 = ppl_train[:-1]  
    
    x1 = list(range(1, len(y1) + 1))
    x2 = list(range(1, len(y2) + 1))
    
    plt.plot(x1, y1, label='PPL dev')
    plt.plot(x2, y2, label='PPL train')
    plt.xlabel('Epochs')
    plt.ylabel('PPL')
    plt.title('PPL')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{filename}.jpg")
    plt.close()

    y1 = losses_dev  # Last value is best_ppl
    y2 = losses_train

    x1 = list(range(1, len(y1) + 1))
    x2 = list(range(1, len(y2) + 1))
    
    plt.plot(x1, y1, label='Loss dev')
    plt.plot(x2, y2, label='Loss train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{filename1}.jpg")
    plt.close()

# Function used to save the model
def save_model(model, filename):
    PATH = os.path.join("bin", filename)
    torch.save(model.state_dict(), PATH)