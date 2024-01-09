import os
import math
import random
import numpy as np
import tensorflow as tf
import glob
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from data_parser import data_generator, fast_data_generator, find_starting_idices, parse_file
import matplotlib
from tensorflow.python.client import device_lib 
from global_variables import *
from zukunftsplots import *

data_path = DATA_PATH_PROCESSED

best_transformer_config = {
    'steps_per_epoch': 100,
    'epochs': 50,
    'seq_len_past': 840,
    'seq_len_future': 300,
    'dropout': 0,
    'batch_size': 1,
    'learning_rate': 0.0001,
    'num_transformer_blocks': 2,
    'mlp_units': [128, 128, 128],
    'model_name': "model-025-mae1.0893.h5"
}

def load_best_transformer(best_transf_config: dict) -> tf.keras.Model:
    # Get the model path
    # model_path = 'models\\' + f"transformer\\{steps_per_epoch}_{epochs}_{seq_len_past}_{seq_len_future}_{dropout}_{batch_size}_{learning_rate}_{num_transformer_blocks}_{mlp_units}"
    model_path = 'best_models\\' + f"{best_transf_config['steps_per_epoch']}_{best_transf_config['epochs']}_{best_transf_config['seq_len_past']}_{best_transf_config['seq_len_future']}_{best_transf_config['dropout']}_{best_transf_config['batch_size']}_{best_transf_config['learning_rate']}_{best_transf_config['num_transformer_blocks']}_{best_transf_config['mlp_units']}"
    model_path = 'best_models\\' + '100_50_840_300_0_4_0.0001_2_[128, 128, 128]'
    if os.path.exists(model_path):
        print("model path exists", model_path)

    # Get the model file path
    # model_file_path = model_path + '\\' + best_transf_config['model_name']

    # h5_files is a list of full paths to the .h5 files. If you want just the file names, you can do:
    model_file_path = model_path + '\\' + best_transf_config["model_name"] # [os.path.basename(file) for file in h5_files][0]

    if os.path.isfile(model_file_path):
        print("model EXISTS!", model_file_path)
        model = tf.keras.models.load_model(model_file_path)
        return model
    else:
        print("model does NOT exist!", model_file_path)
        return None
    
def gen_data(given_years, batch_size, model: tf.keras.Model) -> tuple:
    # Use the last 840 samples to predict the next 300 outputs

    # TODO:
    all_data_dict = get_data_dict(DATA_PATH + CITYS)
    all_data_dict = filter_for_consistent(all_data_dict, 2010, 1, 840)
    random_key = random.choice(list(all_data_dict.keys()))
    # print(len(all_data_dict["ÅrhusDenmark57.05N10.33E"][0]))
    # train_gen = filter_for_consistent()
    x_data, y_data = np.array(all_data_dict[random_key][0]), np.array(all_data_dict[random_key][1])

    # train_gen = data_generator(batch_size, True, True) 
    # x_data, y_data = next(train_gen) # get the next batch
    # print("x data", x_data.shape)
    # print("y data", y_data.shape)

    # Reshape or squeeze x_data and y_data into 1D arrays
    x_data_1d = np.squeeze(x_data)[-(num_given_years * 12):] # only shot last 25 years of given data
    y_data_1d = np.squeeze(y_data)

    y_pred = model.predict(x_data) # reshape x_data into a 1 x 840 array
    # print("prediction:", y_pred.shape)
    # Reshape or squeeze y_pred into a 1D array
    y_pred_1d = np.squeeze(y_pred)

    # print("x data 1d squeez", x_data_1d.shape)
    # print("y data 1d squeez", y_data_1d.shape)
    # print("y pred 1d squeez", y_pred_1d.shape)

    return x_data_1d, y_data_1d, y_pred_1d
    
def average_data(data):
    # Average each 12 entries of x_data_1d
    averages = np.array([np.mean(data[i:i+12]) for i in range(0, len(data), 12)])
    return averages

def plot_stuff(previous_data, actual_data, predicted_data, num_years=25):  
    def save_plot():
        # Check if the directory exists
        if not os.path.exists('./prediction_plots'):
            # If the directory does not exist, create it
            os.makedirs('./prediction_plots')

        # Save the plot as an image file
        plt.savefig('./prediction_plots/transformer_pred.png')
    
    print("previous data", previous_data.shape)
    print("actual data", actual_data.shape)
    print("predicted data", predicted_data.shape)

    # set plot size
    plt.figure(figsize=(15, 6))

    # plt.plot(np.arange(0, num_given_years), x_averages, color='blue', label='Input samples')
    plt.plot(np.arange(0, num_years * 12), previous_data, color='blue', label='Input samples')
        
    # Plot the moving averages
    plt.plot(np.arange(num_years * 12, (num_years * 12) + len(actual_data)), actual_data, color='green', label='actual outputs')
    plt.plot(np.arange(num_years * 12, (num_years * 12) + len(predicted_data)), predicted_data, color='red', label='Predicted outputs')

    # Define your scale
    scale = np.arange(0, (num_years * 12 ) + len(previous_data), 12)
    label = np.arange(0, ((num_years) + (len(actual_data)) / 12)).astype(int)

    # Set the x-axis labels
    plt.xticks(scale, labels=label, rotation=90)

    plt.title(f'Predicted Temperatures for the next {len(previous_data)//12} years')
    plt.xlabel('Years')
    plt.ylabel('Temperatures (°C)')

    plt.legend()

    save_plot()

    plt.show()

#################### Transformer part ####################
num_given_years = 25

# transformer_model = load_best_transformer(best_transformer_config)
# x_data_1d, y_data_1d, y_pred_1d  = gen_data(num_given_years, batch_size=best_transformer_config['batch_size'], model=transformer_model)

# # Average each 12 entries of x_data_1d
# x_averages = average_data(x_data_1d) # np.array([np.mean(x_data_1d[i:i+12]) for i in range(0, len(x_data_1d), 12)])
# y_averages = average_data(y_data_1d) # np.array([np.mean(y_data_1d[i:i+12]) for i in range(0, len(y_data_1d), 12)])
# y_pred_averages = average_data(y_pred_1d) # np.array([np.mean(y_pred_1d[i:i+12]) for i in range(0, len(y_pred_1d), 12)])

# plot_stuff(x_data_1d, y_data_1d, y_pred_1d, num_years=num_given_years)