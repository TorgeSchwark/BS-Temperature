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
import time

class ModelHistory(tf.keras.callbacks.Callback):
  def __init__(self, model_path):
    self.model_path = model_path
    self.loss = []
    self.loss_val = []
    self.mae = []
    self.mae_val = []
    self.mse = []
    self.mse_val = []

  def on_epoch_end(self, batch, logs):
    if  logs.get('loss') < MAX_LOSS:
        self.loss.append(logs.get('loss'))
        self.loss_val.append(logs.get('val_loss'))
        self.mae.append(logs.get('mean_absolute_error'))
        self.mse.append(logs.get('mean_squared_error'))
        self.mae_val.append(logs.get('val_mean_absolute_error'))
        self.mse_val.append(logs.get('val_mean_squared_error'))
        self.plot_data()
    else:
        self.loss.append(MAX_LOSS)
        self.loss_val.append(MAX_LOSS)
        self.mae.append(MAX_LOSS)
        self.mse.append(MAX_LOSS)
        self.mae_val.append(MAX_LOSS)
        self.mse_val.append(MAX_LOSS)
        self.plot_data()
       
  def plot_data(self):
    vis_path = os.path.join(self.model_path, 'vis')

    if not os.path.exists(vis_path):
      os.makedirs(vis_path)

    model_name = self.model_path.split('/')[-1]

    plt.clf() # clear figure
    plt.plot(self.loss)
    plt.plot(self.loss_val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(vis_path, 'loss.png'))

def setup_model_lstm():
  input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

  x = tf.keras.layers.LSTM(500,return_sequences=True)(input)
  x = tf.keras.layers.LSTM(500)(x)
  x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
  x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

  model = tf.keras.models.Model(input, x)
  return model  
 
def setup_model_conv_1d(filters: list = [100], dropout: list = []):
  input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

  dp = 0
  if dropout and len(filters) == len(dropout): # list is not empty
    x = tf.keras.layers.Conv1D(filters=filters[0], kernel_size=5, activation='relu', padding='same')(input)
    x = tf.keras.layers.Dropout(dropout[0])(x)
    for i in range(1, len(filters)):
        x = tf.keras.layers.Conv1D(filters=filters[i], kernel_size=5, activation='relu', padding='same')(input)
        x = tf.keras.layers.Dropout(dropout[i])(x)
  else:
    x = tf.keras.layers.Conv1D(filters=filters[0], kernel_size=5, activation='relu', padding='same')(input)
    x = tf.keras.layers.Dropout(0)(x)
    for i in range(1, len(filters)):
      x = tf.keras.layers.Conv1D(filters=filters[i], kernel_size=5, activation='relu', padding='same')(input)
      x = tf.keras.layers.Dropout(0)(x)
     
  # x = tf.keras.layers.Conv1D(100, kernel_size=5, activation='relu', padding='same')(x)
  # x = tf.keras.layers.Dropout(dp)(x)
  # x = tf.keras.layers.Conv1D(100, kernel_size=5, activation='relu', padding='same')(x)
  # x = tf.keras.layers.Dropout(dp)(x)
  # x = tf.keras.layers.Conv1D(100, kernel_size=3, activation='relu', padding='same')(x)
  # x = tf.keras.layers.Dropout(dp)(x)  
  # x = tf.keras.layers.Conv1D(100, kernel_size=3, activation='relu', padding='same')(x)
  # x = tf.keras.layers.Dropout(dp)(x)  
  x = tf.keras.layers.GlobalAveragePooling1D()(x)

  x = tf.keras.layers.Dense(1500, activation='relu')(x)  
  x = tf.keras.layers.Dense(1500, activation='relu')(x)  
  x = tf.keras.layers.Dense(1000, activation='relu')(x)  
  x = tf.keras.layers.Dense(500, activation='relu')(x)  
  x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
  x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

  model = tf.keras.models.Model(input, x)
  return model  
#architecture := [[layer_size,kernal_size],[layer_size,kernel_size]...]
def setup_variable_conv_1d(architcture):
   hello = 1

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

#Sets up mlp with different variable architecture and dropout
def setup_variable_mlp(architecture,dropout):

    input = tf.keras.layers.Input(shape=(SEQ_LEN_PAST, NUM_INPUT_PARAMETERS), name='input')

    x = tf.keras.layers.Flatten()(input)
    for layer_size in architecture:
        x = tf.keras.layers.Dense(layer_size, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(SEQ_LEN_FUTURE * NUM_OUTPUT_PARAMETERS, activation='linear')(x)
    x = tf.keras.layers.Reshape((SEQ_LEN_FUTURE, NUM_OUTPUT_PARAMETERS))(x)

    model = tf.keras.models.Model(input, x)
    return model

def plot_data_partial(sub_seq_input, sub_seq_label, sub_seq_pred, model_path, epoch, count):
    
    vis_path = os.path.join(model_path, 'vis', 'plots')
    if not os.path.exists(vis_path):
      os.makedirs(vis_path)

    input_samples_y = []
    for i in range(sub_seq_input.shape[1]):
      input_samples_y.append(sub_seq_input[0][i][0])

    label_samples_y = []
    for i in range(sub_seq_label.shape[0]):
      label_samples_y.append(sub_seq_label[i][0])      

    pred_samples_y = []
    for i in range(sub_seq_pred.shape[0]):
      pred_samples_y.append(sub_seq_pred[i][0])      

    plt.clf()
    
    tmp = list(range(0, len(input_samples_y)))
    plt.scatter(tmp, input_samples_y)    
    pred_x = list(range(len(input_samples_y),len(input_samples_y)+len(pred_samples_y)))
    print(pred_x,pred_samples_y)
    plt.scatter(pred_x, pred_samples_y)
    plt.scatter(pred_x, label_samples_y)
    
    plt.title('Sin Plot Partial')
    plt.xlabel('X')
    plt.ylabel('Y')    
    plt.legend(['Inputs', 'Pred', 'Label'], loc='upper left')
    plt.savefig(vis_path + '\\' + str(epoch).zfill(4) + '_' + str(count).zfill(4) + '_p.png')    

# validate model via random chosen samples (depending on validation percentage)
def val_func(data_path, model_path, model, epoch):
    all_files = sorted(glob.glob(data_path + '*.txt'))
    num_plots = 10

    for i in range(num_plots):
        idx_file = random.randint(0,(int((len(all_files)-1)*VALIDATION_PERCENTAGE)))

        samples = parse_file(all_files[idx_file])
        valid_ind = find_starting_idices(samples)

        if(len(valid_ind) != 0):
            idx_seq = random.randint(0,len(valid_ind)-1)
            sub_seq_input = samples[(valid_ind[idx_seq]-SEQ_LEN_FUTURE-SEQ_LEN_PAST+1):(valid_ind[idx_seq]-SEQ_LEN_FUTURE+1)]
            sub_seq_label = samples[valid_ind[idx_seq]-SEQ_LEN_FUTURE+1:valid_ind[idx_seq]+1]
        
        sub_seq_input = np.asarray(sub_seq_input) 
        sub_seq_label = np.asarray(sub_seq_label)
        sub_seq_input = np.expand_dims(sub_seq_input, axis=0) # add new dimension on axis (index) = 0

        preds = model.predict(sub_seq_input)[0]

        plot_data_partial(sub_seq_input, sub_seq_label, preds, model_path, epoch, i)

def train(data_path, model_path, model, batch_size, lr, from_checkpoint=False):

    train_gen = data_generator(batch_size, True) 
    val_gen = data_generator(batch_size, False)

    # TODO: ggf. vary on optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # placeholders: {epoch:03d} -> current epoch number, {val_mae:.4f} -> current validation mean absolute error
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_path + 'model-{epoch:03d}-mae{val_mae:.4f}.h5', verbose=1, monitor='val_mae', save_best_only=True, mode='auto')
    model_history_callback = ModelHistory(model_path)

    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #   monitor='val_loss',
    #   min_delta=0.5,
    #   patience=5,
    #   verbose=1,
    #   mode='auto',
    #   baseline=None,
    #   restore_best_weights=False,
    #   # start_from_epoch=0
    # )
    # needed for Adam optimizer?
    def adapt_learning_rate(epoch):
      return 0.001 * epoch
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(adapt_learning_rate)


    model.compile(loss='mse', optimizer=opt, metrics=["mse", "mae"])
    model.summary()
    
    history = model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, epochs=EPOCHS, 
                      validation_data=val_gen, callbacks=[checkpoint_callback,model_history_callback], #,checkpoint_callback, val_callback, early_stopping_callback],
                      shuffle=True, verbose='auto')
    val_func(data_path, model_path, model, EPOCHS)

# Like normal setup just with Training parameter loop
def lopp_setup():
   
   batch_sizes = [100,200]
   learning_rates = [0.001,0.0005,0.0001]
   Architectures = [[1500,1500,1500,1500,1500],[1500,1500,1500,2000,5000,2000,1000,1000,500],[2000,2000,2000,2000,2000,2000,2000,2000,2000]]
   dropouts = [0]

   with tf.device(GPU_STRING):
        data_path = DATA_PATH # DATA_PATH_PROCESSED ?

        # iterate through multiple hyper parameter configs and train mlp models
        for batch_size in batch_sizes:
           for learning_rate in learning_rates:
              for architecture in Architectures:
                for dropout in dropouts:
                    model_path =  'models\\' + 'models_mlp_spe30_ep25_slp1000_slf360\\' + str(dropout) + str(batch_size) + str(learning_rate) + str(architecture) + '\\'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    
                    model = setup_variable_mlp(architecture,dropout)
                    mode = 'train'
                    train(data_path, model_path, model, batch_size, learning_rate)
          

def loop_setup2():
  batch_sizes = [100,200]
  learning_rates = [0.01,0.001,0.0005]
  # Architectures = [[1500,1500,1500,1500,1500],[1500,1500,1500,2000,5000,2000,1000,1000,500],[2000,2000,2000,2000,2000,2000,2000,2000,2000]]
  Con_Architecture: list = [[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [50, 50, 50, 50, 50], [50, 100, 150, 200, 100], [50, 100, 200, 400, 100]]
  Drop_Architecture: list = [[], [0.3, 0.3, 0.3, 0.3, 0.3], [0.1, 0.1, 0.1, 0.1, 0.1], [0.4, 0.5, 0.5, 0.6, 0.4]]
  dropouts = [0]

  with tf.device(GPU_STRING):
        data_path = DATA_PATH_PROCESSED

        # iterate through multiple hyper parameter configs and train models
        for batch_size in batch_sizes:
           for learning_rate in learning_rates:
              for architecture in Con_Architecture:
                for dropout in Drop_Architecture:
                    # spe = steps per epoch, ep = epochs, slp = sequence len past, slf = sequence len future
                    model_path =  'models_conv\\' + 'models_conv_spe30_ep25_slp1000_slf360\\' + str(dropout) + str(batch_size) + str(learning_rate) + str(architecture) + '\\'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    
                    # model = setup_variable_mlp(architecture,dropout)
                    model = setup_model_conv_1d(architecture,dropout)
                    mode = 'train'
                    train(data_path, model_path, model, batch_size, learning_rate)

                    
  

def normal_setup():
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("\nGPUs: {}\n".format(physical_devices))
    
    with tf.device(GPU_STRING):
       
        data_path = DATA_PATH_PROCESSED
        # model_path = 'models\\' + MODEL_NAME + '\\'
        model_path = 'models\\' + MODEL_NAME + '\\'
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model = setup_model_lstm()

        mode = 'train'

        train(data_path, model_path, model, BATCH_SIZE, 0.001)

def run():
    physical_devices = tf.config.list_physical_devices('GPU')
    print("\nGPUs: {}\n".format(physical_devices))
    
    start = time.time()
    # normal_setup()
    loop_setup2()
    end = time.time()
    print(f"\n Compute time:{end - start}")

if __name__== "__main__":
  run()
