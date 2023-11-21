import math
import random
from global_variables import *
import numpy as np
import glob

import tensorflow as tf
 
# TODO: maybe consider random variance in data to analyse the "robustness" of the model (maybe also augmentation?)

# TODO: --- This should be optimized ---
# helpfull links:
# https://www.tensorflow.org/guide/data

def to_tensor_dataset(x_train: np.ArrayLike, y_train: np.ArrayLike):
    # converting data to tensors
    my_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # <TensorSliceDataset element_spec=(TensorSec(shape=(TODO, TODO), dtype=tf.TODO, name=None), 
    # TensorSpec(shape=(), dtype=tf.TODO, name=None))>
    # ! shape does not contain the amount of data (labels) -> len(my_dataset) = amout of data dimension (e.g. for (28, 28, 6000) it is 6000)
    
    # could be cached: my_dataset = my_dataset.cache() when e.g. maped before
    my_dataset = my_dataset.cache() # can be also done after batching
    for (value, label) in my_dataset:
        # value, labels are tensors
        pass

    my_dataset = my_dataset.shuffle(len(my_dataset)) # maybe not that usefull when getting time/order related data
    my_dataset = my_dataset.batch(32)

    my_dataset = my_dataset.prefetch(tf.data.AUTOTUNE) # run this after all tensor stuff is done to some how speed things up

    # data_generator would simply return yield my_test_dataset my_val_dataset
    # as before where these are tensorflow datasets


def parse_file(file_name):
    file = open(file_name,"r")
    lines = file.readlines()

    samples = []
    count = 0
    for line in lines:
        x, y = line.strip().split(' ')
        samples.append((float(y),))
    return samples

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
        if(is_train): 
            idx_file = random.randint(int((len(all_files)-1)*VALIDATION_PERCENTAGE),int(len(all_files)-1))
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

def data_generator(path: str, batch_size: int, is_train: bool) -> yield:
    all_files = sorted(glob.glob(path + '*.txt'))

    while True:
        inputs, labels = select_data(batch_size, all_files, is_train)
       
        #Data Augmentation
        mu = 0
        sigma = 0.0
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs += rnd

        yield inputs, labels

