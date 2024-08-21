import os
import torch
from conll import evaluate
from sklearn.metrics import classification_report
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import BertTokenizer

# Create a tokenizer using the BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function used to train the model on a training dataset
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    
    # Set the model in training mode
    model.train()
    loss_array = []

    # Loop over the dataset
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        
        # Inference
        slots, intent = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])
        
        # Compute the loss
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        
        loss_array.append(loss.item())
        
        # Backward pass
        loss.backward() 
        
        # Clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        
        # Update the weights
        optimizer.step()

    return loss_array

# Function used to evaluate the model on a validation/test dataset
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    
    # Set the model in evaluation mode
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        
        # Loop over the dataset
        for sample in data:
            # Inference
            slots, intents = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])

            # Compute the loss
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            
            loss_array.append(loss.item())
            
            # Decode the intents
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Decode the slots 
            output_slots = torch.argmax(slots, dim=1)
            
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)

                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                
                tmp_seq = []
                
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                
                hyp_slots.append(tmp_seq)
    
        tmp_ref = []
        tmp_hyp = []
        tmp_ref_tot = []
        tmp_hyp_tot = []

        # Removes the padding ('pad') and special tokens ('[CLS]' and '[SEP]') from both reference and predicted slot sequences to ensure accurate evaluation.
        for ref, hyp in zip(ref_slots, hyp_slots):
            tmp_ref = []
            tmp_hyp = []

            for r, h in zip(ref, hyp):
                if r[1] != 'pad' and r[0] != '[CLS]' and r[0] != '[SEP]':
                    tmp_ref.append(r)
                    tmp_hyp.append(h)
            
            tmp_ref_tot.append(tmp_ref)
            tmp_hyp_tot.append(tmp_hyp)
        
        ref_slots = tmp_ref_tot
        hyp_slots = tmp_hyp_tot

    try:            
        # Evaluate the model
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    
    return results, report_intent, loss_array

# Function used to plot the loss values
def plot_graph(losses_dev, losses_train, filename):

    y1 = losses_dev  
    y2 = losses_train

    x1 = list(range(1, len(y1) + 1))  # indx + 1
    x2 = list(range(1, len(y2) + 1))  # indx + 1
    
    plt.plot(x1, y1, label='Loss dev')
    plt.plot(x2, y2, label='Loss train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{filename}.jpg")
    plt.close()

# Function used to save the model
def save_model(epoch, model, optimizer, lang, filename):
    PATH = os.path.join("exam/248715_pietro_bologna/NLU/part_2/bin", filename)
    saving_object = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'w2id': lang.w2id,
        'slot2id': lang.slot2id,
        'intent2id': lang.intent2id
    }
    torch.save(saving_object, PATH)