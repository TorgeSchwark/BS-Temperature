import os
import random
import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

PC = False

VAL_TO_LEARN_RATION = 0.5

if PC == True:
    DATA_PATH = 'C:\\Users\\Torge\\Desktop\\Uni\\5Semester\\Bachelor Seminar\\Datensatz_Erderwärmung\\dataToProcess\\'
    DATA_PATH_TO = 'C:\\Users\\Torge\\Desktop\\Uni\\5Semester\\Bachelor Seminar\\Datensatz_Erderwärmung\\processed_data\\'
else: 
    DATA_PATH_TO = '/home/torge/Schreibtisch/Git Repos/BS-Temperature/processed_data/'
    DATA_PATH = '/home/torge/Schreibtisch/Git Repos/BS-Temperature/dataToProcess/'

CITYS ='GlobalLandTemperaturesByCity.csv'
COUNTRYS = 'GlobalLandTemperaturesByCountry.csv'
MAJORCITYS = 'GlobalLandTemperaturesByMajorCity.csv'
STATE = 'GlobalLandTemperaturesByState.csv'

def run():
    process_countrys(DATA_PATH + COUNTRYS)
    process_countrys(DATA_PATH + STATE)
    process_citys(DATA_PATH + CITYS)
    process_citys(DATA_PATH + MAJORCITYS)
    
    
def process_countrys(file_to_read):
    df = pd.read_csv(file_to_read)
    old_file_name = ''
    
    for index, row in df.iterrows():
        if file_to_read == DATA_PATH + COUNTRYS:
            file_name = row['Country']
        elif file_to_read == DATA_PATH + STATE:
            file_name = row['Country'] + row['State']

        if file_name != old_file_name:
            old_file_name = file_name
            with open(DATA_PATH_TO + file_name + '.txt', 'w') as file:
                file.write('')
                
        with open(DATA_PATH_TO + file_name + '.txt', 'a') as file:
            temperature = round(float(row['AverageTemperature']), 2)
            datum = row['dt']
            file.write(datum + " ")
            file.write(f'{temperature}\n')


def process_citys(file_to_read):
    df = pd.read_csv(file_to_read)
    old_file_name = ''
    
    for index, row in df.iterrows():
        file_name = row['City'] + row['Country'] + str(row['Latitude']) + str(row['Longitude'])
        
        if file_name != old_file_name:
            old_file_name = file_name
            with open(DATA_PATH_TO + file_name + '.txt', 'w') as file:
                file.write('')  
                
        with open(DATA_PATH_TO + file_name + '.txt', 'a') as file:
            datum = row['dt']
            temperature = round(float(row['AverageTemperature']), 2)
            file.write(datum + " ")
            file.write(str(temperature) + '\n')



if __name__== "__main__":
  run()
