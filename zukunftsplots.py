from global_variables import *
import os
import math
import random
import tensorflow as tf
from data_parser  import get_all_files_as_list
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


model_path = ""
seq_len_past = 840 
seq_len_future = 300 

def get_data_dict(file_to_read):

    df = pd.read_csv(file_to_read)

    city_name = ""
    old_city_name = ""

    all_data_dict = {}
    temp_list = []
    year_list = []

    for index,row in df.head(10000).iterrows():

        city_name = row['City'] + row['Country'] + str(row['Latitude']) + str(row['Longitude'])

        if city_name != old_city_name:
            old_city_name = city_name
            all_data_dict[city_name] = [[],[]]  # [[temperatur1, temp2, ...],[date1, date2, ...]] ; date_format: 1992-02-08
        
        datum = row['dt']
        temperature = round(float(row['AverageTemperature']), 2)
        all_data_dict[city_name][0].append(temperature)
        all_data_dict[city_name][1].append(datum)

    print(len(all_data_dict))

    return all_data_dict
        

def filter_for_consistent(all_data_dict, year, month, lenght):

    new_data_dict = {}
    new_data_dict_with_all_years = {}
    hallo = 1
    for city in all_data_dict:
        count = 0
        for ind_in_city in range(len(all_data_dict[city][0])):
            if not math.isnan(all_data_dict[city][0][ind_in_city]):
                count += 1
            else:
                count = 0
            

            if int(all_data_dict[city][1][ind_in_city][0:4]) == year and int(all_data_dict[city][1][ind_in_city][5:7]) == month and count >= lenght:
                new_data_dict[city] = [all_data_dict[city][0][ind_in_city-lenght+1:ind_in_city+1], 
                                       all_data_dict[city][1][ind_in_city-lenght+1:ind_in_city+1]] 
                new_data_dict_with_all_years[city] = [all_data_dict[city][0], 
                                       all_data_dict[city][1]] 


    return new_data_dict, new_data_dict_with_all_years


def create_avg_diag(all_data_dict, new_data_dict_with_all_years):

    year_dict = {}

    for key in new_data_dict_with_all_years:
        for date_list_ind in range(len(new_data_dict_with_all_years[key][1])):
            year = new_data_dict_with_all_years[key][1][date_list_ind][0:4]
            temperatur = new_data_dict_with_all_years[key][0][date_list_ind]
            if not year in year_dict:
                year_dict[year] = [temperatur]
            else:
                year_dict[year].append(temperatur)
    
    for key in year_dict:
        year_dict[key] = np.nanmean(year_dict[key])

    x_values = list(year_dict.keys())
    y_values = list(year_dict.values())

    plt.plot(x_values[20:], y_values[20:])

    plt.xticks(np.arange(0, len(x_values[20:]), 10), rotation=90)

    # Beschriftungen hinzufügen
    plt.xlabel('Years')
    plt.ylabel('AVG-Temperature')
    plt.title('AVG-Temperature')



def create_future_pots():

    if os.path.exists(model_path):
        print("model found")

    # model = tf.keras.models.load_model(model_path)

    all_data_dict = get_data_dict(DATA_PATH + CITYS)

    all_data_dict, new_data_dict_with_all_years = filter_for_consistent(all_data_dict, 2010, 1, 840)

    create_avg_diag(all_data_dict, new_data_dict_with_all_years)


create_future_pots()

# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (74,) + inhomogeneous part.
