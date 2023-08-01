# run_file

from classifiermod_idsw import TorchNet, get_data
from train_test_loops import train_loop, validation_loop, train_model, hyperparameter_sweep
import torch
import pickle 
from useful_functions import set_device


date = 310723
save_location = r'C:\\Users\\Nay\\pY\\phd_classifer\\saves'
data_file_path = r"C:\Users\Nay\OneDrive\Butterfly_Network\LeedsButterfly2\leedsbutterfly_dataset_v1.1\leedsbutterfly\images"

model = TorchNet()

# files = dl_box_data(items) # download folder of data from box

def main():
    device = set_device()
    #x_train, y_train, x_test, y_test, x_val, y_val = get_data()
    title = 'test_run'
    fresh_model = False
    #train_model(device, model, x_train, y_train, x_val, y_val, save_location, date,title, fresh_model= fresh_model, epochs=2)


if __name__ == "__main__":
    main()