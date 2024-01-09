import os
import math
import random
import numpy as np
import tensorflow as tf
import glob
from matplotlib import pyplot as plt
from data_parser import data_generator, fast_data_generator, find_starting_idices, parse_file
import matplotlib
from tensorflow.python.client import device_lib 
from global_variables import *


data_path = DATA_PATH_PROCESSED
steps_per_epoch = 100
epochs = 50
seq_len_past = 840 # +300 ? due to training ?
seq_len_future = 300 # 40 * 12
dropout = 0
batch_size = 1
learning_rate = 0.0005
num_transformer_blocks = 8
mlp_units=[128, 128]

model_path = 'models\\' + f"transformer\\{steps_per_epoch}_{epochs}_{seq_len_past}_{seq_len_future}_{dropout}_{batch_size}_{learning_rate}_{num_transformer_blocks}_{mlp_units}"
if os.path.exists(model_path):
    print("model path exists")

model_file_path = model_path + '\\model-044-mae1.1406.h5'
if os.path.isfile(model_file_path):
    print("model EXISTS!")
    model = tf.keras.models.load_model(model_file_path)
# Use the last 840 samples to predict the next 300 outputs
train_gen = data_generator(batch_size, True, True) 
x_data, y_data = next(train_gen) # get the next batch
print("x data", x_data.shape)
print("y data", y_data.shape)

num_given_years = 25

# Reshape or squeeze x_data and y_data into 1D arrays
x_data_1d = np.squeeze(x_data)[-(num_given_years * 12):] # only shot last 25 years of given data
print("     -> len:", len(x_data_1d))
y_data_1d = np.squeeze(y_data)
print("     -> len:", len(y_data_1d))

y_pred = model.predict(x_data.reshape(1, -1)) # reshape x_data into a 1 x 840 array
print("prediction:", y_pred.shape)
y_pred_1d = np.squeeze(y_pred)

# # Plot the last 12 samples
# plt.plot(np.arange(24), x_data_1d, color='blue', label='Input samples')

# # Plot the predicted outputs
# plt.plot(np.arange(24, 24 + len(y_data_1d)), y_data_1d, color='red', label='Predicted outputs')

# plt.legend()
# plt.show()

# print(x_data_1d)
# print(y_data_1d)

# Average each 12 entries of x_data_1d
x_averages = np.array([np.mean(x_data_1d[i:i+12]) for i in range(0, len(x_data_1d), 12)])
y_averages = np.array([np.mean(y_data_1d[i:i+12]) for i in range(0, len(y_data_1d), 12)])
y_pred_averages = np.array([np.mean(y_pred_1d[i:i+12]) for i in range(0, len(y_pred_1d), 12)])

# Compute moving averages over each bucket of 12 values
y_data_1d_moving_avg = np.convolve(y_data_1d, np.ones(12), 'valid') / 12

# Plot the last 12 samples
plt.plot(np.arange(0, num_given_years), x_averages, color='blue', label='Input samples')

# Plot the moving averages
# plt.plot(np.arange((num_given_years * 12) // 12, ((num_given_years * 12) // 12) + len(y_averages)), y_averages, color='green', label='actual outputs')
# plt.plot(np.arange((num_given_years * 12) // 12, ((num_given_years * 12) // 12) + len(y_pred_averages)), y_pred_averages, color='red', label='Predicted outputs')
plt.plot(np.arange(num_given_years, num_given_years + (len(y_data_1d) // 12)), y_data_1d, color='green', label='actual outputs')
# plt.plot(np.arange((num_given_years * 12) // 12, ((num_given_years * 12) // 12) + len(y_pred_1d)), y_pred_1d, color='red', label='Predicted outputs')

print("scaling:", np.arange(0, (num_given_years) + len(y_data_1d), 12))
print(" -> len:", len(np.arange(0, (num_given_years) + len(y_data_1d), 12)))
print("labeling:", np.arange(0, ((num_given_years) + len(y_data_1d)) / 12).astype(int))
print(" -> len:", len(np.arange(0, ((num_given_years) + len(y_data_1d)) / 12).astype(int)))

plt.xticks(np.arange(num_given_years+ (len(x_data_1d) // 12) ), np.arange(num_given_years+ (len(x_data_1d) // 12)).astype(int))
# plt.xticks(np.arange(0, (num_given_years) + len(y_data_1d), 12), np.arange(0, ((num_given_years) + len(y_data_1d)) / 12).astype(int))

plt.title('Predicted Temperatures for the next 25 years')
plt.xlabel('Years')
plt.ylabel('Temperatures (°C)')

plt.legend()

# Save the plot as an image file
plt.savefig('./resources/prediction_plot.png')

plt.show()