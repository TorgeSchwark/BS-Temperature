import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.colors import LightSource

import os
import re

# model_path =  'models\\' + 'models_lstm_rest\\' + str(dropout) + str(batch_size) + str(learning_rate) + str(architecture) + str(STEPS_PER_EPOCH) +str(EPOCHS) +str(SEQ_LEN_PAST)+ str(SEQ_LEN_FUTURE)+ '\\'
# model-{epoch:03d}-mae{val_mae:.4f}.h5'

def extract_parameters_from_path(path):
    model_name: str = path.split("\\")[1]
    # print("\n!!!!!!!!!!! model:", model_name)
    if (model_name == "models_lstm"):
        # print("!!!!!!!!!!! model models_lstm\n")
        match = re.match(r"models\\models_lstm\\(\d+)_(\d+\.\d+)_(\[.*?\])_(\d+)_(\d+)_(\d+)_(\d+)\\", path)
        if match:
            batch_size, learning_rate, architecture_str, steps_per_epoch, epochs, seq_len_past, seq_len_future = match.groups()
            # architecture = [int(x) for x in architecture_str.replace(" ", "").split(",")]
            architecture = eval(architecture_str) 
            # Store the parameters in a dictionary
            parameters_dict = {
                'architecture_name': "models_lstm",
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'architecture': architecture,
                'steps_per_epoch': steps_per_epoch,
                'epochs': epochs,
                'seq_len_past': seq_len_past,
                'seq_len_future': seq_len_future
            }
            # print("got params: ", parameters_dict)
            return parameters_dict
    elif (model_name == "models_lstm_rest"):
        # print("!!!!!!!!!!! model models_lstm_rest\n")
        match = re.match(r"models\\models_lstm_rest\\(\d+)_(\d+)_(\d+\.?\d*(?:[eE][-+]?\d+)?)_\[([,\d\s]+)\]_(\d+)_(\d+)_(\d+)_(\d+)", path)
        if match:
            dropout, batch_size, learning_rate, architecture, steps_per_epoch, epochs, seq_len_past, seq_len_future = match.groups()
            architecture = [int(x) for x in architecture.replace(" ", "").split(",")]
            # Store the parameters in a dictionary
            parameters_dict = {
                'architecture_name': "models_lstm_rest",
                'dropout': dropout,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'architecture': architecture,
                'steps_per_epoch': steps_per_epoch,
                'epochs': epochs,
                'seq_len_past': seq_len_past,
                'seq_len_future': seq_len_future
            }
            # print("got params: ", parameters_dict)
            return parameters_dict
    elif (model_name == "models_conv_spe"):
        # print("!!!!!!!!!!! model models_conv_spe\n")
        match = re.match(r'models\\models_conv_spe\\(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\[.*?\])\\', path)
        if match:
            steps_per_epoch, epochs, seq_len_past, seq_len_future, batch_size, learning_rate, architecture_str = match.groups()
            # architecture = [int(x) for x in architecture_str.replace(" ", "").split(",")]
            architecture = eval(architecture_str)
            # Store the parameters in a dictionary
            parameters_dict = {
                'architecture_name': "models_conv_spe",
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'architecture': architecture,
                'steps_per_epoch': steps_per_epoch,
                'epochs': epochs,
                'seq_len_past': seq_len_past,
                'seq_len_future': seq_len_future
            }
            # print("got params: ", parameters_dict)
            return parameters_dict
    else:
        pass

    return None

def extract_parameters_from_model_name(model_name) -> dict:
    # Extract parameters from model name using regular expressions
    match = re.match(r'model-(\d+)-mae(\d+\.\d+)', model_name)
    if match:
        epoch, val_mae = match.groups()
        parameters = {
            'epoch': int(epoch),
            'val_mae': float(val_mae)
        }
        # print("     model params: ", parameters)
        return parameters
    else:
        return None

def iterate_and_extract_parameters(root_folder):
    parameter_list = []

    for model_type in os.listdir(root_folder):
        type_path = os.path.join(root_folder, model_type)
        # print("type path:" + str(type_path) + " -> is a dir", os.path.isdir(type_path))
        if os.path.isdir(type_path):
            for model_config in os.listdir(type_path):
                model_config_path = os.path.join(type_path, model_config)
                # print("     model_config_path:" + str(model_config) + " -> is a dir", os.path.isdir(model_config_path))
                if os.path.isdir(model_config_path):
                    for file_name in os.listdir(model_config_path):
                        if file_name.endswith('.h5'):
                            file_path = os.path.join(model_config_path, file_name)
                            # print("     extract info from:", file_path)
                            model_parameters = extract_parameters_from_model_name(file_name)
                            path_parameters = extract_parameters_from_path(file_path)
                            
                            if path_parameters and model_parameters:
                                parameters = {**path_parameters, **model_parameters}
                                parameter_list.append(parameters)
                else: # maybe older model storing methods
                    for file_name in os.listdir(type_path):
                        if file_name.endswith('.h5'):
                            file_path = os.path.join(model_config_path, file_name)
                            # print("     file:", file_name)
                            model_parameters = extract_parameters_from_model_name(file_name)
                            path_parameters = extract_parameters_from_path(file_path)
                            
                            if path_parameters and model_parameters:
                                parameters = {**path_parameters, **model_parameters}
                                parameter_list.append(parameters)
    return parameter_list

# Example usage:
root_folder = 'models'
print("DOIN Stuff")
parameters_list = iterate_and_extract_parameters(root_folder)

print("\n\nRESULTS")
# Display the extracted parameters
for parameters in parameters_list:
    print(parameters)

####
## ERGEBNISSE CONV -----------------------------------------------
###

batch_sizes = []
learning_rates = []
mae = []
current_config = ""
config_num = 0
for res in parameters_list:
    # print(f"batch_sizes: {batch_sizes}\nlearning_rate: {learning_rates}\nmae: {mae}\n config_num = {config_num}")

    if res["architecture_name"] == "models_conv_spe": 
        if current_config == "": 
            current_config = f"{res['batch_size']}_{res['learning_rate']}_{res['architecture_name']}_{res['steps_per_epoch']}_{res['epochs']}"
            batch_sizes.append([res["batch_size"]])
            learning_rates.append([res["learning_rate"]])
            mae.append([res["val_mae"]])
        else:
            if (current_config == f"{res['batch_size']}_{res['learning_rate']}_{res['architecture_name']}_{res['steps_per_epoch']}_{res['epochs']}"):
                batch_sizes[config_num].append(res["batch_size"])
                learning_rates[config_num].append(res["learning_rate"])
                mae[config_num].append(res["val_mae"])
            else:
                config_num += 1
                current_config = f"{res['batch_size']}_{res['learning_rate']}_{res['architecture_name']}_{res['steps_per_epoch']}_{res['epochs']}"
                batch_sizes.append([res["batch_size"]])
                learning_rates.append([res["learning_rate"]])
                mae.append([res["val_mae"]])

print(f"\nbatch_sizes: {batch_sizes}\nlearning_rate: {learning_rates}\nmae: {mae}")

# flipping lists - for testing reasons
batch_sizes = batch_sizes[::-1][0][:3]
# batch_sizes = [elem[0] for elem in batch_sizes]
learning_rates = learning_rates[::-1]
learning_rates = [0.0001,0.001,0.002,0.0004]
# learning_rates = [elem[0] for elem in learning_rates]
mae = mae[::-1][0][:3]
# mae = [elem[0] for elem in mae]

# batch_sizes = [[10, 10, 10], [5, 5, 5]]
# learning_rates = [[0.001, 0.002, 0.003], [0.005, 0.006, 0.007]]
# mae = [[1.2, 1.5, 1.8], [2.0, 2.3, 2.6]]

print(f"\nbatch_sizes: {batch_sizes}\nlearning_rate: {learning_rates}\nmae: {mae}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes[0], learning_rates[0], mae[0], linewidth=0.2, antialiased=True, color='darkblue')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE ')
ax.set_title('MAE [[10, 10, 10,], [5, 5, 5]] arch')

plt.show()
plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surface1 = ax.plot_trisurf(batch_sizes[1], learning_rates[1], mae[1], linewidth=0.2, antialiased=True, color='darkred')

# ax.set_xlabel('Batchsize')
# ax.set_ylabel('Learning rate')
# ax.set_zlabel('MAE ')
# ax.set_title('MAE [[20, 20, 20, 20, 20], [30, 30, 30, 30, 30]] arch')

# plt.show()
# plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surface1 = ax.plot_trisurf(batch_sizes[2], learning_rates[2], mae[2], linewidth=0.2, antialiased=True, color='green')

# ax.set_xlabel('Batchsize')
# ax.set_ylabel('Learning rate')
# ax.set_zlabel('MAE ')
# ax.set_title('MAE [[50, 50, 50, 50, 50], [10, 20, 20, 20, 10]] arch')

# plt.show()
# plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surface1 = ax.plot_trisurf(batch_sizes[3], learning_rates[3], mae[3], linewidth=0.2, antialiased=True, color='grey')

# ax.set_xlabel('Batchsize')
# ax.set_ylabel('Learning rate')
# ax.set_zlabel('MAE ')
# ax.set_title('MAE [[50, 50, 50], [10, 10, 10]] arch')

# plt.show()
# plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surface1 = ax.plot_trisurf(batch_sizes[0], learning_rates[0], mae[0], linewidth=0.2, antialiased=True, color='darkblue',shade=False, alpha=0.5)
# surface2 = ax.plot_trisurf(batch_sizes[1], learning_rates[1], mae[1], linewidth=0.2, antialiased=True, color='darkred',shade=False, alpha=0.5)
# surface3 = ax.plot_trisurf(batch_sizes[2], learning_rates[2], mae[2], linewidth=0.2, antialiased=True, color='green',shade=False, alpha=0.55)
# surface3 = ax.plot_trisurf(batch_sizes[3], learning_rates[3], mae[3], linewidth=0.2, antialiased=True, color='grey',shade=False, alpha=0.6)

# ax.set_xlabel('Batchsize')
# ax.set_ylabel('Learning rate')
# ax.set_zlabel('MAE')
# ax.set_title('compared arch')

# plt.show()
# plt.close()