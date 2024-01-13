import math
import random
from global_variables import *
import numpy as np
import glob
from multiprocessing import Pool


from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

scaler = MinMaxScaler()


# TODO: maybe consider random variance in data to analyse the "robustness" of the model (maybe also augmentation?)

# TODO: --- This should be optimized ---
# helpfull links:
# https://www.tensorflow.org/guide/data

def to_tensor_dataset(x_train, y_train): # np.Array
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

# gerne einmal anschauen, dass sollte so funktionieren.
# count zählt wie viele valid daten nacheinander gefunden wurden
# ind_in_file und ind_in_files_list sind einfach die indizes 



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
    

def generate_clipped_random_value(mu, sigma, lower_bound, upper_bound):
    while True:
        # Generiere einen zufälligen Wert aus der Normalverteilung
        random_value = np.random.normal(mu, sigma)

        # Überprüfe, ob der Wert innerhalb des Bereichs liegt
        if lower_bound <= random_value <= upper_bound:
            return random_value
        
def generate_clipped_gaussian_value(mu, sigma, lower_bound, upper_bound):
    while True:
        # Generiere einen zufälligen Wert aus der Normalverteilung
        random_value = np.random.normal(mu, sigma)

        # Überprüfe, ob der Wert innerhalb des Bereichs liegt
        if lower_bound <= random_value <= upper_bound:
            return random_value

def select_data(batch_size, all_files, is_train, augmentation):
    selected_inputs = []
    selected_labels = []
    
    valid_files = all_files[int((len(all_files) - 1) * VALIDATION_PERCENTAGE):] if is_train else all_files[:int((len(all_files) - 1) * VALIDATION_PERCENTAGE)]

    for _ in range(batch_size):
        idx_file = random.choice(valid_files)
        samples = parse_file(idx_file)
        valid_ind = find_starting_idices(samples)

        if len(valid_ind) != 0:
            idx_seq = random.choice(valid_ind)
            
            sub_seq_input = np.array(samples[(idx_seq - SEQ_LEN_FUTURE - SEQ_LEN_PAST + 1):(idx_seq - SEQ_LEN_FUTURE + 1)])
            sub_seq_label = np.array(samples[idx_seq - SEQ_LEN_FUTURE + 1: idx_seq + 1])

            if augmentation:
                random_multip = generate_clipped_random_value(1.25, 0.25, 0.5, 2.0)
                sub_seq_input *= random_multip
                sub_seq_label *= random_multip

                random_offset = generate_clipped_gaussian_value(0, 5, -10, 10)
                sub_seq_input += random_offset
                sub_seq_label += random_offset

            selected_inputs.append(sub_seq_input)
            selected_labels.append(sub_seq_label)

    
    return np.asarray(selected_inputs), np.asarray(selected_labels)


def generate_data(args):
    batch_size, is_train, augmentation, all_files = args
    inputs, labels = select_data(batch_size, all_files, is_train, augmentation)

    return inputs, labels


def data_generator(batch_size, is_train, augmentation, pool, num_processes):
    path = DATA_PATH_PROCESSED
    all_files = sorted(glob.glob(path + '*.txt'))

    while True:
        args_list = [(batch_size, is_train, augmentation, all_files) for _ in range(num_processes)]
        results = pool.map(generate_data, args_list)

        for result in results:
            yield result




def get_all_files_as_list():
    # files_list contains all files as lists eg: [[1,2,3,4,5,5],[1,2,3,4,5,5],[1,2,3,4,5,5]]
    # valid_ind_dict contains valid indices in file considering Input_seq_len and output_seq_len
    # in (float(temperature),) format for data parsing 
    files_list = []
    ind_in_files_list = 0
    ind_in_file = 0
    file_list = []
    valid_ind_dict = {}
    all_files = sorted(glob.glob(DATA_PATH_PROCESSED + '*.txt'))
    for file_name in all_files:
        file_list = []
        count = 0
        ind_in_file = 0
        valid_ind_dict[ind_in_files_list] = []

        file = open(file_name,"r")
        lines = file.readlines()
        for line in lines:
            temperature = float(line.strip().split(' ')[1])
            file_list.append((float(temperature),))
            if math.isnan(temperature):
                count = 0
            if count >= SEQ_LEN_FUTURE+SEQ_LEN_PAST:
                valid_ind_dict[ind_in_files_list].append(ind_in_file)
            ind_in_file += 1
            count += 1
            
        ind_in_files_list += 1
        files_list.append(file_list)
        
    return files_list, valid_ind_dict



def information():
    data, dict = get_all_files_as_list()
    count = 0
    for country in data:
        count += len(country)
    print(count)


information()