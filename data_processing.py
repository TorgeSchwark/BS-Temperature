from global_variables import *
import pandas as pd
import os

#This Takes a while ! about 20 min 
#necesary for fair data collection

def run_data():
    if not os.path.exists(DATA_PATH_PROCESSED):
        os.makedirs(DATA_PATH_PROCESSED)
    
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
            
            with open(DATA_PATH_PROCESSED + file_name + '.txt', 'w') as file:
                file.write('')
                
        with open(DATA_PATH_PROCESSED + file_name + '.txt', 'a') as file:
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
            with open(DATA_PATH_PROCESSED + file_name + '.txt', 'w') as file:
                file.write('')  
                
        with open(DATA_PATH_PROCESSED + file_name + '.txt', 'a') as file:
            datum = row['dt']
            temperature = round(float(row['AverageTemperature']), 2)
            file.write(datum + " ")
            file.write(str(temperature) + '\n')

if __name__== "__main__":
  run_data()
