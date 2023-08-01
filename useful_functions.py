# useful_functions
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.nn import functional
import datetime
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle

# label distribution histogram function
def label_dist_plot(data, label):
  plt.figure()

  data = [int(x) for x in data]
  _, _, _ = plt.hist(data, bins=[0, 1, 2,3,4,5,6,7,8,9,10,11], align='left') # bins = number of classes
  plt.xticks(np.unique(data))
  plt.xlim(left=min(np.unique(data))-1, right=max(np.unique(data))+1)
  plt.title(label)
  plt.show()

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device


# image to tensor function
def tensoring(input, device):
        input = cv2.imread(input)
        input = cv2.resize(input, (64, 64))
        input = input.astype('float32')
        #input = input/255
        img_tensor = torch.tensor(input)
        img_tensor = img_tensor.to(torch.float32)
        img_tensor = functional.normalize(img_tensor)
        img_tensor = img_tensor.permute(2, 0, 1)    ######### <<--------- permute is needed for multichannel data. set for 3 channels.
        img_tensor = img_tensor.reshape(1, 3, 64, 64)
        img_tensor = img_tensor.to(device)

        return img_tensor

# label one hot encodeing function
def label_oh_tf(lab, device):
    num_classes = 11
    one_hot = np.zeros(num_classes)
    lab = int(lab)
    one_hot[lab] = 1
    label = torch.tensor(one_hot)
    label = label.to(torch.float32)
    label = label.to(device) #
    return label





execution = datetime.datetime.now()

# learning curve
def learning_curve(v_loss_list, t_loss_list, save_location):
  plt.title(label="Learning Curve", fontsize =30)
  plt.plot(range(len(t_loss_list)), t_loss_list, label ='Training loss')
  plt.plot(range(len(v_loss_list)), v_loss_list, label='Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  #plt.yscale("log")
  plt.legend()
  plt.savefig(save_location+'/learningCuve'+'finForestRun'+'lr1e-4_'+'wd1e-5_'+'ep30_'+str(execution)+'.png') #run_name
  plt.show()
  # save figs

# accuracy curve
def accuracy_curve(v_accuracy_list, t_accuracy_list, save_location):
  plt.title(label="Accuracy Curve", fontsize =30)
  plt.plot(range(len(t_accuracy_list)), t_accuracy_list, label ='Training accuracy')
  plt.plot(range(len(v_accuracy_list)), v_accuracy_list, label='Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(save_location+'/accuracyCuve'+'finForestRun'+'lr1e-4_'+'wd1e-5_'+'ep30_'+'.png', format='png')
  plt.show()


  # confusion matrix


def confusion_matrix(label_list, predict_list, title:str): #  
    epoch_in_question =-1
    labels = [int(y) for y in label_list]
    final_pred = [x.cpu() for x in predict_list[epoch_in_question]]

    print(title)
    train_epoch_matrix = confusion_matrix(labels, final_pred)
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9,10])
    disp.plot()
    plt.show()


# load saved pickle model
def load_pickle(file_name:str, model, device, save_location):
    model.to(device)
    #load_epoch = -1

    with open(save_location+f''+file_name, 'rb') as f:
        save_dict = pickle.load(f)

    # access dictionary elements
    #run_name = save_dict['Run']
    #total_epochs = save_dict['Current Epoch']
    model_state_dict = save_dict['model.state_dict']
    #training_samples = save_dict['training_samples']
    #validation_samples = save_dict['validation_samples']

    #t_loss_list = save_dict['t_loss_list']
    #t_predict_list = save_dict['t_predict_list']
    #t_accuracy_list = save_dict['t_accuracy_list']

    #v_loss_list = save_dict['v_loss_list']
    #v_predict_list = save_dict['v_predict_list']
    #v_accuracy_list = save_dict['v_accuracy_list']

    model.load_state_dict(model_state_dict)
    return model, save_dict

def reset_lists(date, title):
  t_loss_list = []
  t_predict_list =[]
  t_accuracy_list = []

  v_loss_list = []
  v_predict_list =[]
  v_accuracy_list = []

  total_epochs = 0

  title = f'{title}_{date}'
  save_dict = {'Run' : title,
              'Current_Epoch': 0}
  return t_loss_list, t_predict_list, t_accuracy_list, v_loss_list, v_predict_list, v_accuracy_list, total_epochs, title, save_dict