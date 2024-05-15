
    n_epochs = 100
    patience = 3 #prevent overfitting and save computational power
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    cut_epochs = []
    pbar = tqdm(range(1,n_epochs))

    ppl_train_array = []
    ppl_dev_array = []

    weights_update = {}
    best_weights = {}
    switch_optimizer = False

    stored_loss = math.inf
    best_val_loss = []
    hyp_control_monotonic = 5
    counting_weight = 0

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        ppl, loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_train_array.append(ppl)
        losses_train.append(np.asarray(loss).mean())
        sampled_epochs.append(epoch)

        if epoch % 1 == 0:
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppl_dev_array.append(ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())

            pbar.set_description("PPL: %f" % ppl_dev)

            
            if  ppl_dev < best_ppl: 
                best_ppl = ppl_dev 
                best_model = copy.deepcopy(model).to('cpu')
                for parameter in model.parameters():
                    best_weights[parameter] = parameter.data.clone()
                    #saving the parameter of the best model for using them to restart in that point
                
                patience = 3
            elif ppl_dev > best_ppl and switch_optimizer: #if the model is not improving but the optimazer is switched, the patience is decreased
                patience -= 1
                lr = lr / 2

            if patience <= 0: # and switch_optimizer: # Early stopping with patience
                break # Not nice but it keeps the code clean

            #with this if we control if is the case so switch the optimizer (SGD to ASGD)
            if switch_optimizer == False and (len(losses_dev) > hyp_control_monotonic and losses_dev > min(best_val_loss[:-hyp_control_monotonic])):
                switch_optimizer = True 
                weights_update = best_weights 
            
            elif switch_optimizer:
                counting_weight += 1
                tmp = {}
                for parameter in model.parameters():
                    tmp[parameter] = parameter.data.clone()
                    weights_update[parameter] = tmp[parameter]
                    
                    average = weights_update[parameter] / counting_weight
                    parameter.data = average.data.clone()
                 

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    
    # To save the model
    name = 'model_LSTM_21'
    path = 'bin/' + name + '.pt'
    torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))

    #-----------------------#
    path_info = 'PART_21/'
    save_infos (path_info, name, lr, hid_size, emb_size, losses_train, losses_dev, ppl_train_array, ppl_dev_array, sampled_epochs, final_ppl, False)
    #-----------------------#