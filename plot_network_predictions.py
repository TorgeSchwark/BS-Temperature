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
# print(x_data.shape)
# print(y_data.shape)

# Reshape or squeeze x_data and y_data into 1D arrays
x_data_1d = np.squeeze(x_data)[-24:]
y_data_1d = np.squeeze(y_data)

# y_pred = model.predict(x_data.reshape(1, -1))
# print(y_pred.shape)

# Plot the last 12 samples
plt.plot(np.arange(24), x_data_1d, color='blue', label='Input samples')

# Plot the predicted outputs
plt.plot(np.arange(24, 24 + len(y_data_1d)), y_data_1d, color='red', label='Predicted outputs')

plt.legend()
plt.show()

print(x_data_1d)
print(y_data_1d)

# Average the first 12 entries of x_data_1d
# avg_first_12 = np.mean(x_data_1d[:12])

# Average the second 12 entries of x_data_1d
# avg_second_12 = np.mean(x_data_1d[12:])

# print(avg_first_12)
# print(avg_second_12)

# Compute moving averages over each bucket of 12 values
y_data_1d_moving_avg = np.convolve(y_data_1d, np.ones(12), 'valid') / 12

# Plot the last 12 samples
plt.plot(np.arange(12), x_data_1d, color='blue', label='Input samples')

# Plot the moving averages
plt.plot(np.arange(12, 12 + len(y_data_1d_moving_avg)), y_data_1d_moving_avg, color='red', label='Predicted outputs')

plt.legend()
plt.show()