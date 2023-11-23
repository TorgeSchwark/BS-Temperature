import math
import random
from global_variables import *
import numpy as np
import glob

# gerne einmal anschauen, dass sollte so funktionieren.
# count zählt wie viele valid daten nacheinander gefunden wurden
# ind_in_file und ind_in_files_list sind einfach die indizes 

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
            ind_in_file += 1
            count += 1

            temperature = float(line.strip().split(' ')[1])
            file_list.append((float(temperature),))
            if math.isnan(temperature):
                count = 0
            if count >= SEQ_LEN_FUTURE+SEQ_LEN_PAST:
                valid_ind_dict[ind_in_files_list].append(ind_in_file)
        ind_in_files_list += 1
        files_list.append(file_list)
    return files_list, valid_ind_dict



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

def fast_select_data(data, ind_dict, batch_size, is_train):
    selected_inputs = []
    selected_labels = []

    num = 0
    while num < batch_size:
        if is_train:
            indx_file = random.randint(int((len(data)-1)*VALIDATION_PERCENTAGE),len(data)-1)
        else:
            indx_file = random.randint(0,int((len(data)-1)*VALIDATION_PERCENTAGE))
        
        if len(ind_dict[indx_file]) != 0:
            indx_seq = random.randint(0,len(ind_dict[indx_file])-1)
            sub_seq_input = data[indx_file][(ind_dict[indx_file][indx_seq]-SEQ_LEN_FUTURE-SEQ_LEN_PAST+1):(ind_dict[indx_file][indx_seq]-SEQ_LEN_FUTURE+1)]
            sub_seq_label = data[indx_file][ind_dict[indx_file][indx_seq]-SEQ_LEN_FUTURE+1:ind_dict[indx_file][indx_seq]+1]

            selected_inputs.append(sub_seq_input)
            selected_labels.append(sub_seq_label)

            num += 1

    return np.asarray(selected_inputs), np.asarray(selected_labels)

def data_generator( batch_size, is_train):
    path = DATA_PATH_PROCESSED
    all_files = sorted(glob.glob(path + '*.txt'))

    while True:
        inputs, labels = select_data(batch_size, all_files, is_train)
       
        #Data Augmentation
        mu = 0
        sigma = 0.0
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs += rnd
        yield inputs, labels

def fast_data_generator(batch_size, is_train):
    data, ind_dict = get_all_files_as_list()
    
    while True:
        inputs, labels = fast_select_data(data, ind_dict, batch_size, is_train)

        #Data Augmentation
        mu = 0
        sigma = 0.0
        rnd = np.random.normal(mu, sigma, size=inputs.shape)
        inputs += rnd

        yield inputs, labels



def test():
    all_files = sorted(glob.glob(DATA_PATH_PROCESSED + '*.txt'))
    data, dict = get_all_files_as_list()
    samples = parse_file(all_files[0])

    print(data[0])
    print("haaaaaloooooooooooooooooooooooooooooo \n")
    print(samples)

test()
