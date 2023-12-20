from global_variables import *
import pandas as pd
import os
import json


def read_files():

    df = pd.read_csv(DATA_PATH+CITYS)
    old_file_name = ''
    data_dict = {}
    for index, row in df.iterrows():
        file_name = row['City'] + row['Country'] + str(row['Latitude']) + str(row['Longitude'])
        
        data_list = []
        if file_name != old_file_name:
            old_file_name = file_name
            data_dict[file_name] = []
        
        data_dict[file_name].append(round(float(row['AverageTemperature']), 2))

    return data_dict

def search_for_duplicates():
    data_dict = read_files()
    same_count = 0
    duplicate_cities = []
    gleiche_cities_gruppen = []
    for key in data_dict:
        gleiche_cities_gruppe = [key]
        key_in_duplicates = False
        if not key in duplicate_cities:
            for key2 in data_dict:
                if data_dict[key] == data_dict[key2] and key != key2:
                    if not key_in_duplicates:
                        duplicate_cities.append(key)
                        key_in_duplicates = True
                    gleiche_cities_gruppe.append(key2)
                    duplicate_cities.append(key2)
            if len(gleiche_cities_gruppe) > 1:
                same_count += len(gleiche_cities_gruppe)
                gleiche_cities_gruppen.append(gleiche_cities_gruppe)

    print("es gibt :", same_count , "gleiche städte aus ", len(data_dict))
    dateipfad = 'duplicate_cities.json'
    if not os.path.exists(DUPLICATES):
        os.makedirs(DUPLICATES)
    with open(DUPLICATES+dateipfad, 'w') as datei:
        json.dump(gleiche_cities_gruppen, datei)

    print(f'Daten wurden in {DUPLICATES} gespeichert.')
            




search_for_duplicates()