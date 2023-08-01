# train, val & test loops
import torch
from useful_functions import tensoring, label_oh_tf, set_device, reset_lists
import pickle
from IPython.display import clear_output
from classifiermod_idsw import TorchNet

# train loop
def train_loop(model, x_train, y_train, epoch, optimizer, loss_fn):
  model = model
  x_train = x_train
  y_train = y_train

  model.train()

  predict_list = []
  total_count = 0
  num_correct = 0
  current_loss = 0

  total_samples = len(x_train)

  for idx, img in enumerate(x_train):

      tense = tensoring(img)
      prediction = model.forward(tense)
      label = label_oh_tf(y_train[idx])

      loss = loss_fn(prediction, label)
      predict_list.append(prediction.argmax())

      #print('\n ---------------------------------------------------------------')
      #print('             Epoch: ', epoch, '  Sample: ', idx)

      if prediction.argmax() == label.argmax():
          print(f'\n ########################### HIT ###########################  -- {idx} / {total_samples} \n')
          num_correct +=1
      else:
        #print('\n ########################### MISS ########################### \n')
        pass

      total_count+=1

      #print(prediction, '\n Prediction:  ', prediction.argmax())
      #print('Label: ',label.argmax())
      #print('Loss: ', loss.item())
      #print('---------------------------------------------------------------')
      #print(" |||| ||||| ||||| ||||| ||||| ||||| |||| |||| ||||| |||| |||| ")

      current_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


  return current_loss, predict_list, num_correct, model, optimizer



# validation loop

def validation_loop(model, x_val, y_val, epoch, loss_fn):
  model = model
  x_val = x_val
  y_val = y_val

  model.eval()

  predict_list = []
  total_count = 0
  num_correct = 0
  current_loss = 0

  for idx, img in enumerate(x_val):

      tense_img = tensoring(img)
      prediction = model.forward(tense_img)
      label = label_oh_tf(y_val[idx])

      loss = loss_fn(prediction, label)
      predict_list.append(prediction.argmax())

      #print('\n ---------------------------------------------------------------')
      #print('             Epoch: ', epoch, '  Sample: ', idx)

      if prediction.argmax() == label.argmax():
          #print('\n ########################### HIT ########################### \n')
          num_correct +=1
      else:
        #print('\n ########################### MISS ########################### \n')
        pass
      total_count+=1

      #print('ArgPrediction: ', prediction.argmax()) #, prediction,'ARRRGGGG',
      #print('Label: ',label.argmax())
      #print('Loss: ', loss.item())
      #print('---------------------------------------------------------------')
      #print(" |||| ||||| ||||| ||||| ||||| ||||| |||| |||| ||||| |||| |||| ")

      current_loss += loss.item()

  return current_loss, predict_list, num_correct


def train_model(device, model, x_train, y_train, x_val, y_val, save_location, date,title, fresh_model = True, epochs= 1):
    if model == None or fresh_model == True:
      model =TorchNet().to(device)
      t_loss_list, t_predict_list, t_accuracy_list, v_loss_list, v_predict_list, v_accuracy_list, total_epochs, title, save_dict = reset_lists(date=date, title=title)
      
    # settings
    # number of epochs you want to run 
    total_epochs = total_epochs # important for previously made models
    learning_rate= 1e-4   #5e-5
    loss_fn = torch.nn.MSELoss()   #torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  #, weight_decay=1e-5)#, weight_decay=1e-5)
    optim= 'Adam' # for recording model parameters

    """
    print('\n')
    print('LR: ', learning_rate)
    print('optimiser: ', optimizer)
    print('loss fn: ', loss_fn)
    """
  

    for epoch in range(epochs):
        #print('lr: ',learning_rate, 'optim: ',optim, 'loss fn: ',loss_fn)
        print('EPOCH: ', epoch)
        print('----------------------')
        print(' \n                  TRAINING... \n')
        print('----------------------')
        train_loss, train_predict_loss, train_num_correct, model, optimizer = train_loop(model, x_train, y_train, epoch, optimizer, loss_fn)
        t_loss_list.append(train_loss)
        t_predict_list.append(train_predict_loss)
        t_accuracy_list.append(train_num_correct / len(y_train))
        final_train_acc = t_accuracy_list[-1]

        print('----------------------')
        print(' \n                  VALIDATION... \n')
        print('----------------------')
        val_loss, val_predict_loss, val_num_correct = validation_loop(model, x_val, y_val, epoch, loss_fn)
        v_loss_list.append(val_loss)
        v_predict_list.append(val_predict_loss)
        v_accuracy_list.append(val_num_correct/ len(y_val))
        final_val_acc = v_accuracy_list[-1]

        total_epochs += 1 ### Total epochs is from a previous save

        # save model data in dict
        save_dict['Current Epoch'] = total_epochs
        save_dict['model.state_dict'] = model.state_dict()
        save_dict['training_samples'] = len(x_train)
        save_dict['validation_samples'] = len(x_val)
        save_dict['t_loss_list'] = t_loss_list
        save_dict['t_predict_list'] = t_predict_list
        save_dict['t_accuracy_list'] = t_accuracy_list
        save_dict['v_loss_list'] = v_loss_list
        save_dict['v_predict_list'] = v_predict_list
        save_dict['v_accuracy_list'] = v_accuracy_list
        #save_dict['epochCount']+=1 Now using current_epoch above

        if epoch == epochs-1:  ### Mabs change this to something like epoch == epochs, so that it only saves the final?
            version = f'extra64lay_epoch{total_epochs}_lr{str(learning_rate)}_{optim}_{str(loss_fn)}+_Acc_{final_train_acc}_{final_val_acc}'
            # save dict 
            with open(f'{save_location}/_IDSWforest_epoch{epoch}_{date}_{version}.pkl', 'wb') as f:
                pickle.dump(save_dict, f)



        clear_output()





def hyperparameter_sweep(device, model, x_train, y_train, x_val, y_val, save_location, date):
    # a function to loop through hyperparameters for finding the best ones for certain architecture

    epochs = 30

    lr_list = [1e-4]    #[1e-6, 5e-3, 5e-4, 1e-4, 1e-5] #1e-7, 1e-6, #1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 5e-4, 5e-3, 5e-2
    optimiser_list = ['Adam'] #, 'SGD'
    lossfn_list = [torch.nn.MSELoss()] #, torch.nn.NLLLoss(), ,

    best_optim = None
    best_lossfn = None
    best_lr = 0
    best_valaccuracy = 0
    best_epoch = 0

    for loss_fn in lossfn_list:
        for optim in optimiser_list:
            for learning_rate in lr_list:



                model = model.to(device) #model architecture
                t_loss_list = []
                t_predict_list =[]
                t_accuracy_list = []

                v_loss_list = []
                v_predict_list =[]
                v_accuracy_list = []

                total_epochs = 0
                title = f'forest_colab_HyperParameterSweep_{str(learning_rate)}_Adam_wd30plus_{str(loss_fn)}lecun'
                save_dict = {'Run' : title,
                            'Current_Epoch': 0}
                optim_list=[]
                if optim =='Adam':
                    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
                    optim_list.append(optimizer1)
                    optim_list.append(optimizer2)
                """elif optim == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)"""
                for optimizer in optim_list:


                    print('\n')
                    print('LR: ', learning_rate)
                    print('optimiser: ', optim)
                    print('loss fn: ', loss_fn)

                    for epoch in range(epochs):
                      print('lr: ',learning_rate, 'optim: ',optim, 'loss fn: ',loss_fn)
                      print('EPOCH: ', epoch)
                      print('----------------------')
                      print(' \n                  TRAINING... \n')
                      print('----------------------')
                      train_loss, train_predict_loss, train_num_correct, model, optimizer = train_loop(model, x_train, y_train, epoch, optimizer, loss_fn)
                      t_loss_list.append(train_loss)
                      t_predict_list.append(train_predict_loss)
                      t_accuracy_list.append(train_num_correct / len(y_train))


                      print('----------------------')
                      print(' \n                  VALIDATION... \n')
                      print('----------------------')
                      val_loss, val_predict_loss, val_num_correct = validation_loop(model, x_val, y_val, epoch, loss_fn)
                      v_loss_list.append(val_loss)
                      v_predict_list.append(val_predict_loss)
                      v_accuracy_list.append(val_num_correct/ len(y_val))

                      if v_accuracy_list[-1] > best_valaccuracy:
                            best_valaccuracy = v_accuracy_list[-1]
                            best_optim = optimizer
                            best_lossfn = loss_fn
                            best_lr = learning_rate
                            best_epoch = epoch

                      total_epochs += 1 ### Total epochs is from a previous save


                      save_dict['Current Epoch'] = total_epochs
                      save_dict['model.state_dict'] = model.state_dict()
                      save_dict['training_samples'] = len(x_train)
                      save_dict['validation_samples'] = len(x_val)
                      save_dict['t_loss_list'] = t_loss_list
                      save_dict['t_predict_list'] = t_predict_list
                      save_dict['t_accuracy_list'] = t_accuracy_list
                      save_dict['v_loss_list'] = v_loss_list
                      save_dict['v_predict_list'] = v_predict_list
                      save_dict['v_accuracy_list'] = v_accuracy_list
                      #save_dict['epochCount']+=1 Now using current_epoch above

                      final_train_acc = round(t_accuracy_list[-1],3)
                      final_val_acc = round(v_accuracy_list[-1],3)
                      version =f'forest_v15_epoch{total_epochs}_lr{str(learning_rate)}_{optim}_{str(loss_fn)}_Acc_{final_train_acc}_{final_val_acc}'

                      if epoch==epochs-1:
                          with open(save_location+f'/forest_colab_final_{date}_{version}.pkl', 'wb') as f:
                            pickle.dump(save_dict, f)



                      clear_output()


    print('Top results from hyperparameter sweep:')
    print()
    print(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch)
    return best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch, v_loss_list, t_loss_list, v_accuracy_list, t_accuracy_list

