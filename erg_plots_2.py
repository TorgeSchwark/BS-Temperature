import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.colors import LightSource

import os
import re

# model_path =  'models\\' + 'models_lstm_rest\\' + str(dropout) + str(batch_size) + str(learning_rate) + str(architecture) + str(STEPS_PER_EPOCH) +str(EPOCHS) +str(SEQ_LEN_PAST)+ str(SEQ_LEN_FUTURE)+ '\\'
# model-{epoch:03d}-mae{val_mae:.4f}.h5'


def extract_parameters_from_folder(folder_path):
    # Extract parameters from folder path using regular expressions
    match = re.match(r'models\\models_lstm_rest\\(\[\d+(,\d+)*\])(\d+)(\.\d+)(\[\d+(,\d+)*\])(\d+)(\d+)(\d+)(\d+)', folder_path)
    print("match: ", match)
    if match:
        dropout_list, batch_size, learning_rate, architecture_list, steps_per_epoch, epochs, seq_len_past, seq_len_future = match.groups()

        # Convert the comma-separated strings to lists
        dropout = [int(d) for d in dropout_list.strip('[]').split(',')]
        architecture = [int(a) for a in architecture_list.strip('[]').split(',')]

        parameters = {
            'dropout': dropout,
            'batch_size': int(batch_size),
            'learning_rate': float(learning_rate),
            'architecture': architecture,
            'steps_per_epoch': int(steps_per_epoch),
            'epochs': int(epochs),
            'seq_len_past': int(seq_len_past),
            'seq_len_future': int(seq_len_future)
        }
        return parameters
    else:
        return None

def extract_parameters_from_model_name(model_name) -> dir:
    # Extract parameters from model name using regular expressions
    match = re.match(r'model-(\d+)-mae(\d+\.\d+)', model_name)
    
    if match:
        epoch, val_mae = match.groups()
        parameters = {
            'epoch': int(epoch),
            'val_mae': float(val_mae)
        }
        return parameters
    else:
        return None

def iterate_and_extract_parameters(root_folder = ".\\"):
    parameter_list = []
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        print("folder path:", folder_path)
        if os.path.isdir(folder_path):
            print("     -> is a dir")
            parameters_from_folder = extract_parameters_from_folder(folder_name)
            print("     -> params:", parameters_from_folder)
            if parameters_from_folder:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.h5'):
                        model_parameters = extract_parameters_from_model_name(file_name)
                        if model_parameters:
                            parameters = {**parameters_from_folder, **model_parameters} # merge both dicts into one
                            parameter_list.append(parameters)

    return parameter_list

# Example usage:
root_folder = 'models\\models_lstm_rest'
print("DOIN Stuff")
parameters_list = iterate_and_extract_parameters(root_folder)

# Display the extracted parameters
for parameters in parameters_list:
    print(parameters)

# TODO: do real stuff with plotting

