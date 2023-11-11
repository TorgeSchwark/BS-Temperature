import os
import math
import random
import numpy as np
import tensorflow as tf
import glob
from matplotlib import pyplot as plt
import matplotlib
from tensorflow.python.client import device_lib 

GPU_STRING = '/gpu:0'
BATCH_SIZE = 400
MODEL_NAME = "Versuch1"
EPOCHS = 300
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 32
SEQ_LEN_PAST = 350
SEQ_LEN_FUTURE = 120
NUM_INPUT_PARAMETERS = 1
NUM_OUTPUT_PARAMETERS = 1

VALIDATION_PERCENTAGE = 0.5

def setup_model_mlp():
  
  input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

  dp = 0.0
  x = tf.keras.layers.Flatten()(input)
  x = tf.keras.layers.Dense(1024, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(1024, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(1024, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)
  x = tf.keras.layers.Dense(2000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(2000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(2000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(2000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(5000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(5000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(4000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(2000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(2000, activation='selu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(2000, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(528, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(228, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
  x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

  model = tf.keras.models.Model(input, x)
  return model

def parse_file(file_name):
    file = open(file_name,"r")
    lines = file.readlines()

    samples = []
    count = 0
    for line in lines:
        x, y = line.strip().split(' ')
        samples.append((float(y),))
    return samples

#find all the ind of every year we have consistend data for a valid period
def find_starting_idices(samples):
    count = 0
    valid_ind = []
    for ind_y in range(len(samples)):
        count += 1
        if math.isnan(samples[ind_y][0]):
            count = 0
        if(count >= SEQ_LEN_PAST+SEQ_LEN_FUTURE):
            valid_ind.append(ind_y)
    return valid_ind
    
    

def select_data(batch_size, all_files, is_train):
    selected_inputs = []
    selected_labels = []

    num = 0
    while num < batch_size:
        if(is_train): #we use one part of the data for training and the other for validation
            idx_file = random.randint(0,(int((len(all_files)-1)*(1-VALIDATION_PERCENTAGE))))
        else:
            idx_file = random.randint(0,(int((len(all_files)-1)*VALIDATION_PERCENTAGE)))

        samples = parse_file(all_files[idx_file])
        valid_ind = find_starting_idices(samples)

        if(len(valid_ind) != 0):

            idx_seq = random.randint(0,len(valid_ind)-1)
            sub_seq_input = samples[(valid_ind[idx_seq]-SEQ_LEN_FUTURE-SEQ_LEN_PAST+1):(valid_ind[idx_seq]-SEQ_LEN_FUTURE+1)]
            sub_seq_label = samples[valid_ind[idx_seq]-SEQ_LEN_FUTURE+1:valid_ind[idx_seq]+1]

            selected_inputs.append(sub_seq_input)
            selected_labels.append(sub_seq_label)

            num += 1

    return np.asarray(selected_inputs), np.asarray(selected_labels)

        

def data_generator(path, batch_size, is_train):
    all_files = sorted(glob.glob(path + '*.txt'))

    while True:
        inputs, labels = select_data(batch_size, all_files, is_train)
       
        #Data Augmentation
        mu = 0
        sigma = 0.0
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs += rnd

        yield inputs, labels



def train(data_path, model_path, model, from_checkpoint=False):

    train_gen = data_generator(data_path, BATCH_SIZE, True) 
    val_gen = data_generator(data_path, BATCH_SIZE, False)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00002)

    model.compile(loss='mse', optimizer=opt, metrics=["mse", "mae"])
    model.summary()
    
    history = model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, epochs=EPOCHS, 
                      validation_data=val_gen, callbacks=[],#,checkpoint_callback, val_callback], 
                      shuffle=True, verbose='auto')

def run():
    physical_devices = tf.config.list_physical_devices('GPU')
    print("\n")
    print("GPUs:", physical_devices)
    print("\n")
    
    with tf.device(GPU_STRING):
        path = 'C:\\Users\\Torge\\Desktop\\Uni\\5Semester\\Bachelor Seminar\\Datensatz_Erderwärmung\\'
        data_path = path + 'processed_data\\'
        model_path = path + 'models\\' + MODEL_NAME + '\\'
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model = setup_model_mlp()

        mode = 'train'

        train(data_path, model_path, model)


if __name__== "__main__":
  run()