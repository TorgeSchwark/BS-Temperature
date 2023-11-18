import math
import random
from global_variables import *
import numpy as np
import glob
 
#--- This should be optimized ---

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

