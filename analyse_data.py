import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import numpy as np
from global_variables import *
import os
import csv


LONGITUDE = "Longitude"
LATITUDE = "Latitude"
CONTINENTS = "Continents"
TEMPERATURE = "Temperatur"
YEAR = "Years"
UNCERTAINTY = "Uncertainty"
UNCERTAINTY_PER_YEAR = "Uncertainty_per_year"
DATA_GAPS = "Data_gaps"


country_to_continent = {
    'Denmark': 'Europe',
    'Turkey': 'Asia',
    'Kazakhstan': 'Asia',
    'China': 'Asia',
    'Spain': 'Europe',
    'Germany': 'Europe',
    'Nigeria': 'Africa',
    'Iran': 'Asia',
    'Russia': 'Europe',
    'Canada': 'North America',
    "Côte D'Ivoire": 'Africa',
    'United Kingdom': 'Europe',
    'Saudi Arabia': 'Asia',
    'Japan': 'Asia',
    'United States': 'North America',
    'India': 'Asia',
    'Benin': 'Africa',
    'United Arab Emirates': 'Asia',
    'Mexico': 'North America',
    'Venezuela': 'South America',
    'Ghana': 'Africa',
    'Ethiopia': 'Africa',
    'Australia': 'Australia',
    'Yemen': 'Asia',
    'Indonesia': 'Asia',
    'Morocco': 'Africa',
    'Pakistan': 'Asia',
    'France': 'Europe',
    'Libya': 'Africa',
    'Burma': 'Asia',
    'Brazil': 'South America',
    'South Africa': 'Africa',
    'Syria': 'Asia',
    'Egypt': 'Africa',
    'Algeria': 'Africa',
    'Netherlands': 'Europe',
    'Malaysia': 'Asia',
    'Portugal': 'Europe',
    'Ecuador': 'South America',
    'Italy': 'Europe',
    'Uzbekistan': 'Asia',
    'Philippines': 'Asia',
    'Madagascar': 'Africa',
    'Chile': 'South America',
    'Belgium': 'Europe',
    'El Salvador': 'North America',
    'Romania': 'Europe',
    'Peru': 'South America',
    'Colombia': 'South America',
    'Tanzania': 'Africa',
    'Tunisia': 'Africa',
    'Turkmenistan': 'Asia',
    'Israel': 'Asia',
    'Eritrea': 'Africa',
    'Paraguay': 'South America',
    'Greece': 'Europe',
    'New Zealand': 'Australia',
    'Vietnam': 'Asia',
    'Cameroon': 'Africa',
    'Iraq': 'Asia',
    'Afghanistan': 'Asia',
    'Argentina': 'South America',
    'Azerbaijan': 'Asia',
    'Moldova': 'Europe',
    'Mali': 'Africa',
    'Congo (Democratic Republic Of The)': 'Africa',
    'Thailand': 'Asia',
    'Central African Republic': 'Africa',
    'Bosnia And Herzegovina': 'Europe',
    'Bangladesh': 'Asia',
    'Switzerland': 'Europe',
    'Equatorial Guinea': 'Africa',
    'Cuba': 'North America',
    'Lebanon': 'Asia',
    'Mozambique': 'Africa',
    'Serbia': 'Europe',
    'Angola': 'Africa',
    'Somalia': 'Africa',
    'Norway': 'Europe',
    'Nepal': 'Asia',
    'Poland': 'Europe',
    'Ukraine': 'Europe',
    'Guinea Bissau': 'Africa',
    'Malawi': 'Africa',
    'Burkina Faso': 'Africa',
    'Slovakia': 'Europe',
    'Congo': 'Africa',
    'Belarus': 'Europe',
    'Gambia': 'Africa',
    'Czech Republic': 'Europe',
    'Hungary': 'Europe',
    'Burundi': 'Africa',
    'Zimbabwe': 'Africa',
    'Bulgaria': 'Europe',
    'Haiti': 'North America',
    'Puerto Rico': 'North America',
    'Sri Lanka': 'Asia',
    'Nicaragua': 'North America',
    'Zambia': 'Africa',
    'Honduras': 'North America',
    'Taiwan': 'Asia',
    'Bolivia': 'South America',
    'Guinea': 'Africa',
    'Ireland': 'Europe',
    'Senegal': 'Africa',
    'Latvia': 'Europe',
    'Qatar': 'Asia',
    'Albania': 'Europe',
    'Tajikistan': 'Asia',
    'Kenya': 'Africa',
    'Guatemala': 'North America',
    'Finland': 'Europe',
    'Sierra Leone': 'Africa',
    'Sweden': 'Europe',
    'Botswana': 'Africa',
    'Guyana': 'South America',
    'Austria': 'Europe',
    'Uganda': 'Africa',
    'Armenia': 'Asia',
    'Dominican Republic': 'North America',
    'Jordan': 'Asia',
    'Papua New Guinea': 'Australia',
    'Cambodia': 'Asia',
    'Sudan': 'Africa',
    'Djibouti': 'Africa',
    'Estonia': 'Europe',
    'Panama': 'North America',
    'Mauritius': 'Africa',
    'Jamaica': 'North America',
    'Lesotho': 'Africa',
    'Laos': 'Asia',
    'Bahamas': 'North America',
    'Mauritania': 'Africa',
    'Liberia': 'Africa',
    'Mongolia': 'Asia',
    'Reunion': 'Africa',
    'Lithuania': 'Europe',
    'Macedonia': 'Europe',
    'Togo': 'Africa',
    'Uruguay': 'South America',
    'Chad': 'Africa',
    'Rwanda': 'Africa',
    'Gabon': 'Africa',
    'South Korea': 'Asia',
    'Cyprus': 'Europe',
    'Suriname': 'South America',
    'Slovenia': 'Europe',
    'Oman': 'Asia',
    'Costa Rica': 'North America',
    'Montenegro': 'Europe',
    'Singapore': 'Asia',
    'Bahrain': 'Asia',
    'Iceland': 'Europe',
    'Georgia': 'Asia',
    'Hong Kong': 'Asia',
    'Niger': 'Africa',
    'Namibia': 'Africa',
    'Swaziland': 'Africa',
    'Croatia': 'Europe'
}

def country_to_shortcut(country_name):
    country_mapping = {
        "Andorra": "ad",
        "United Arab Emirates": "ae",
        "Afghanistan": "af",
        "Albania": "al",
        "Armenia": "am",
        "Angola": "ao",
        "Antarctica": "aq",
        "Argentina": "ar",
        "Austria": "at",
        "Australia": "au",
        "Azerbaijan": "az",
        "Bosnia and Herzegovina": "ba",
        "Bangladesh": "bd",
        "Belgium": "be",
        "Burkina Faso": "bf",
        "Bulgaria": "bg",
        "Bahrain": "bh",
        "Burundi": "bi",
        "Benin": "bj",
        "Brunei Darussalam": "bn",
        "Bolivia, Plurinational State of": "bo",
        "Brazil": "br",
        "Bhutan": "bt",
        "Botswana": "bw",
        "Belarus": "by",
        "Belize": "bz",
        "Canada": "ca",
        "Congo, the Democratic Republic of the": "cd",
        "Central African Republic": "cf",
        "Congo": "cg",
        "Switzerland": "ch",
        "Cote d’Ivoire": "ci",
        "Chile": "cl",
        "Cameroon": "cm",
        "China": "cn",
        "Colombia": "co",
        "Costa Rica": "cr",
        "Cuba": "cu",
        "Cape Verde": "cv",
        "Cyprus": "cy",
        "Czech Republic": "cz",
        "Germany": "de",
        "Djibouti": "dj",
        "Denmark": "dk",
        "Dominican Republic": "do",
        "Algeria": "dz",
        "Ecuador": "ec",
        "Estonia": "ee",
        "Egypt": "eg",
        "Western Sahara": "eh",
        "Eritrea": "er",
        "Spain": "es",
        "Ethiopia": "et",
        "Finland": "fi",
        "France": "fr",
        "Gabon": "ga",
        "United Kingdom": "gb",
        "Georgia": "ge",
        "French Guiana": "gf",
        "Ghana": "gh",
        "Greenland": "gl",
        "Gambia": "gm",
        "Guinea": "gn",
        "Equatorial Guinea": "gq",
        "Greece": "gr",
        "Guatemala": "gt",
        "Guam": "gu",
        "Guinea-Bissau": "gw",
        "Guyana": "gy",
        "Hong Kong": "hk",
        "Honduras": "hn",
        "Croatia": "hr",
        "Haiti": "ht",
        "Hungary": "hu",
        "Indonesia": "id",
        "Ireland": "ie",
        "Israel": "il",
        "India": "in",
        "Iraq": "iq",
        "Iran, Islamic Republic of": "ir",
        "Iceland": "is",
        "Italy": "it",
        "Jamaica": "jm",
        "Jordan": "jo",
        "Japan": "jp",
        "Kenya": "ke",
        "Kyrgyzstan": "kg",
        "Cambodia": "kh",
        "Korea, Democratic People’s Republic of": "kp",
        "Korea, Republic of": "kr",
        "Kuwait": "kw",
        "Kazakhstan": "kz",
        "Lao People’s Democratic Republic": "la",
        "Lebanon": "lb",
        "Liechtenstein": "li",
        "Sri Lanka": "lk",
        "Liberia": "lr",
        "Lesotho": "ls",
        "Lithuania": "lt",
        "Luxembourg": "lu",
        "Latvia": "lv",
        "Libyan Arab Jamahiriya": "ly",
        "Morocco": "ma",
        "Monaco": "mc",
        "Moldova, Republic of": "md",
        "Montenegro": "me",
        "Madagascar": "mg",
        "Macedonia, the former Yugoslav Republic of": "mk",
        "Mali": "ml",
        "Myanmar": "mm",
        "Mongolia": "mn",
        "Macao": "mo",
        "Mauritania": "mr",
        "Malta": "mt",
        "Mauritius": "mu",
        "Maldives": "mv",
        "Malawi": "mw",
        "Mexico": "mx",
        "Malaysia": "my",
        "Mozambique": "mz",
        "Namibia": "na",
        "Niger": "ne",
        "Nigeria": "ng",
        "Nicaragua": "ni",
        "Netherlands": "nl",
        "Norway": "no",
        "Nepal": "np",
        "New Zealand": "nz",
        "Oman": "om",
        "Panama": "pa",
        "Peru": "pe",
        "Papua New Guinea": "pg",
        "Philippines": "ph",
        "Pakistan": "pk",
        "Poland": "pl",
        "Puerto Rico": "pr",
        "Palestine, State of": "ps",
        "Portugal": "pt",
        "Paraguay": "py",
        "Reunion": "re",
        "Romania": "ro",
        "Serbia": "rs",
        "Russian Federation": "ru",
        "Rwanda": "rw",
        "Saudi Arabia": "sa",
        "Seychelles": "sc",
        "Sudan": "sd",
        "Sweden": "se",
        "Singapore": "sg",
        "Saint Helena, Ascension and Tristan da Cunha": "sh",
        "Slovenia": "si",
        "Slovakia": "sk",
        "Sierra Leone": "sl",
        "San Marino": "sm",
        "Senegal": "sn",
        "Somalia": "so",
        "Suriname": "sr",
        "Sao Tome and Principe": "st",
        "El Salvador": "sv",
        "Syrian Arab Republic": "sy",
        "Swaziland": "sz",
        "Chad": "td",
        "Togo": "tg",
        "Thailand": "th",
        "Tajikistan": "tj",
        "Timor-Leste": "tl",
        "Turkmenistan": "tm",
        "Tunisia": "tn",
        "Turkey": "tr",
        "Taiwan (Republic of China)": "tw",
        "Tanzania, United Republic of": "tz",
        "Ukraine": "ua",
        "Uganda": "ug",
        "United States": "us",
        "Uruguay": "uy",
        "Uzbekistan": "uz",
        "Holy See (Vatican City State)": "va",
        "Venezuela, Bolivarian Republic of": "ve",
        "Viet Nam": "vn",
        "Yemen": "ye",
        "Mayotte": "yt",
        "South Africa": "za",
        "Zambia": "zm",
        "Zimbabwe": "zw"
    }

    return country_mapping.get(country_name, None)

def create_longitute_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
    df['Longitude'] = df['Longitude'].str.extract('(\d+\.\d+)').astype(float)*(-1+ (df['Longitude'].str.extract('([A-Z]+)')=="E")*2)

    plt.hist(df['Longitude'], bins=36, color='blue', edgecolor='black')

    plt.title('Histogramm of Longitude')
    plt.xlabel('Longitude')
    plt.ylabel('Anzahl')
    plt.tight_layout()

    
    if HISTOGRAMS+LONGITUDE:
        plt.savefig(HISTOGRAMS+LONGITUDE)  

    plt.show()

def create_latitude_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
    df['Latitude'] = df['Latitude'].str.extract('(\d+\.\d+)').astype(float)*(-1+ (df['Latitude'].str.extract('([A-Z]+)')=="N")*2)

    plt.hist(df['Latitude'], bins=18, color='blue',edgecolor='black')

    plt.title('Histogramm of Latitude')
    plt.xlabel('Latitude')
    plt.ylabel('Anzahl')
    plt.tight_layout()
    
    if HISTOGRAMS+LATITUDE:
        plt.savefig(HISTOGRAMS+LATITUDE)  

    plt.show()

def create_continent_barchar(data_path,cities):
    df = pd.read_csv(data_path+cities)
    country_dict = {}
    continent_dict = {}
    #initation of Countrys
    # dict([country_dict[country] = 0 for country in df['Country']]) # also possible as list-dict comprehension :P
    for country in df['Country']:
        country_dict[country] = 0
    #counting datapoints per Country
    for country in df['Country']:
        country_dict[country] += 1
    #initiation of continents
    for country in country_to_continent:
        continent_dict[country_to_continent[country]] = 0
    #summing up the datapoints per countinent
    for country in country_dict:
        continent_dict[country_to_continent[country]] += country_dict[country]

    sum = 0
    for continent in continent_dict:
        sum += continent_dict[continent]
    
    continents = list(continent_dict.keys())
    values = list(continent_dict.values())

    plt.bar(continents, values, color='blue', edgecolor='black')
    plt.title('Data amount by Continent')
    plt.xlabel('Continent')
    plt.xticks(rotation=45)
    plt.ylabel('Amount of Data')
    plt.tight_layout()

    if HISTOGRAMS+CONTINENTS:
        plt.savefig(HISTOGRAMS+CONTINENTS)  

    plt.show()

def create_country_map(data_path,cities):
    df = pd.read_csv(data_path+cities)
    country_dict = {}
    continent_dict = {}
    #initation of Countrys
    # dict([country_dict[country] = 0 for country in df['Country']]) # also possible as list-dict comprehension :P
    for country in df['Country']:
        country_dict[country] = 0
    #counting datapoints per Country
    for country in df['Country']:
        country_dict[country] += 1
    #initiation of continents
    for country in country_to_continent:
        continent_dict[country_to_continent[country]] = 0
    #summing up the datapoints per countinent
    for country in country_dict:
        continent_dict[country_to_continent[country]] += country_dict[country]

    sum = 0
    for continent in continent_dict:
        sum += continent_dict[continent]
    
    continents = list(continent_dict.keys())
    values = list(continent_dict.values())
    print("Count countries:", len(country_dict))
    print("Count continents:", len(continent_dict))

    # create a world map 
    worldmap =  pygal.maps.world.World()
    
    # set the title of the map 
    worldmap.title = 'Countries'
    
    country_counts: dict = {country_to_shortcut(country): value for country, value in country_dict.items()}

    # adding the countries 
    worldmap.add('My Data', country_counts) 
    
    # save into the file 
    worldmap.render_to_file('abc.svg') 
    print("Success") 

# TODO: colour the bins (left = cold -> dark blue, right = hot -> dark red)
def create_temperature_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
    # count temperatures
    temp_dict = {}
    for temp in df['AverageTemperature']:
        if not math.isnan(temp):
            temp_dict[str(int(temp))]  = 0
    for temp in df['AverageTemperature']:
        if not math.isnan(temp):
            temp_dict[str(int(temp))]  += 1

    sorted_data = dict(sorted(temp_dict.items(), key=lambda item: int(item[0])))

    values = list(sorted_data.values())
    labels = list(sorted_data.keys())

    plt.hist(df['AverageTemperature'], bins=90, color='blue', edgecolor='black')

    plt.xticks(rotation=90)
    plt.title('Histogramm of Temperatur')
    plt.xlabel('Temperatur')
    plt.ylabel('Anzahl')
    plt.tight_layout()

    if HISTOGRAMS+TEMPERATURE:
        plt.savefig(HISTOGRAMS+TEMPERATURE)  
    
    plt.show()

# def create_temperature_histogram(data_path, cities):
#     df = pd.read_csv(data_path + cities)

#     # Count temperatures
#     temp_dict = {}
#     for temp in df['AverageTemperature']:
#         if not math.isnan(temp):
#             temp_dict[str(int(temp))] = 0
#     for temp in df['AverageTemperature']:
#         if not math.isnan(temp):
#             temp_dict[str(int(temp))] += 1

#     sorted_data = dict(sorted(temp_dict.items(), key=lambda item: int(item[0])))

#     values = list(sorted_data.values())
#     labels = list(sorted_data.keys())

#     # Normalize temperature values to range [0, 1]
#     normalized_temps = (df['AverageTemperature'] - df['AverageTemperature'].min()) / (df['AverageTemperature'].max() - df['AverageTemperature'].min())

#     # Create a color gradient from dark blue to dark red
#     color_gradient = sns.color_palette("RdYlBu_r", as_cmap=True)(normalized_temps)

#     # Create a histogram using Seaborn with manually set colors for each bar
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(df['AverageTemperature'], height=1, color=color_gradient, edgecolor='black')

#     plt.xticks(rotation=90)
#     plt.title('Histogram of Temperatur')
#     plt.xlabel('Temperatur')
#     plt.ylabel('Anzahl')
#     plt.tight_layout()

#     # Create a colorbar for reference
#     sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(vmin=df['AverageTemperature'].min(), vmax=df['AverageTemperature'].max()))
#     sm.set_array([])  # You need to set a dummy array for the scalar mappable
#     cbar = plt.colorbar(sm)
#     cbar.set_label('Temperatur')

#     # Save or show the plot
#     if HISTOGRAMS + TEMPERATURE:
#         plt.savefig(HISTOGRAMS + TEMPERATURE)
#     else:
#         plt.show()

def create_year_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)

    df['dt'] = pd.to_datetime(df['dt']).dt.year

    plt.hist(df['dt'], bins=271, color='blue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Histogramm of Temperatur')
    plt.xlabel('Temperatur')
    plt.ylabel('Amount of data')
    plt.tight_layout()

    if HISTOGRAMS+YEAR:
        plt.savefig(HISTOGRAMS+YEAR)  
    
    plt.show()

def create_uncertainty_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)

    plt.hist(df['AverageTemperatureUncertainty'], bins=100, color='blue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Histogramm of Temperatur')
    plt.xlabel('Uncetainty')
    plt.ylabel('Amount of data')
    plt.tight_layout()

    if HISTOGRAMS+UNCERTAINTY:
        plt.savefig(HISTOGRAMS+UNCERTAINTY)  
    
    plt.show()

def create_uncertainty_and_nans_per_year_barchar(data_path,cities):
    df = pd.read_csv(data_path+cities) 
    df['Year'] = pd.to_datetime(df['dt']).dt.year
    year_dict = {}
    nan_per_year_dict = {}

    for index, row in df.iterrows():
        year = row['Year']
        avg_temp_uncertainty = row['AverageTemperatureUncertainty']
        
        if not math.isnan(avg_temp_uncertainty):
            if year in year_dict:
                year_dict[year].append(avg_temp_uncertainty)
            else:
                year_dict[year] = [avg_temp_uncertainty]
        else: 
            if year in nan_per_year_dict:
                nan_per_year_dict[year] += 1
            else:
                nan_per_year_dict[year] = 1

    for year in year_dict:
        year_dict[year] = sum(year_dict[year]) / len(year_dict[year])

    print(year_dict)
    plt.bar(year_dict.keys(), year_dict.values(), color='blue')
    plt.xlabel('Year')
    plt.ylabel('AverageTemperatureUncertainty')
    plt.title('AverageTemperatureUncertainty per year')
    plt.tight_layout()

    if HISTOGRAMS+UNCERTAINTY_PER_YEAR:
        plt.savefig(HISTOGRAMS+UNCERTAINTY_PER_YEAR)

    plt.show()

    plt.bar(nan_per_year_dict.keys(),nan_per_year_dict.values(), color='blue')
    plt.xlabel('Year')
    plt.ylabel('Data gaps')
    plt.title('Data gaps per year')
    plt.tight_layout()

    if HISTOGRAMS+DATA_GAPS:
        plt.savefig(HISTOGRAMS+DATA_GAPS)

    plt.show()

# TODO: analysing duplicate / copyed values of different cities (maybe naghbouring)
def analyse_same_val_cities():
    # returns the entries that are suspiciously near to each other and have the exact same avgTemperature
    def distance(same_temp_entries: list) -> list: # (bool, list)
        def parse_coordinates(coord_str: str) -> float:
            """ Parse coordinates from a string with letters indicating cardinal directions.
            Parameters:
                coord_str (str): String representation of coordinates with letters (e.g., '57.05N', '10.33E').
            Returns:
                float: Numerical value of the coordinate."""
            numerical_value, direction = float(coord_str[:-1]), coord_str[-1]
            return numerical_value if direction in ['N', 'E'] else -numerical_value
        # allowed distance between two cities before no sus factor possible (in km)
        TOLERANCE_DIS: int = 50
        # Radius of the Earth in kilometers
        R = 6371.0    
        sus_list: list = []
        for index, dt, city, country, lat, long in same_temp_entries:
            for index2, dt2, city2, country2, lat2, long2 in same_temp_entries:
                if city != city2:                 
                    # Haversine formula
                    lat_val, long_val = parse_coordinates(lat), parse_coordinates(long)
                    lat_val2, long_val2 = parse_coordinates(lat2), parse_coordinates(long2)

                    dlat = lat_val2 - lat_val
                    dlon = long_val2 - long_val
                    
                    a = math.sin(dlat / 2)**2 + math.cos(lat_val) * math.cos(lat_val2) * math.sin(dlon / 2)**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                    # distance in km
                    distance = R * c
                    if distance <= TOLERANCE_DIS:
                        sus_list.append((index, dt, city, country, lat, long, index2, dt2, city2, country2, lat2, long2))
        return sus_list
    
    data_path = DATA_PATH + CITYS
    df = pd.read_csv(data_path)
    same_temp_values: dict = {} # key = temperature, value = [city1, city2, ...]
    # i = 5000
    print("before same_temp_value creation")
    for index, dt, avgTemp, avgTempunc, city, country, lat, long in df.itertuples():
        # if i <= 0: break
        if avgTemp in same_temp_values:
            same_temp_values[avgTemp].append((index, dt, city, country, lat, long))
        else:
            same_temp_values[avgTemp] = [(index, dt, city, country, lat, long)]
        # i -= 1
    print("before distance loop")
    for k, v in same_temp_values.items():
        if len(v) > 1:
            sus_list: list = distance(v)
            if sus_list:
                print("avgTemp:" + str(k) + " sus entries: \n" + str(sus_list))
    pass

def summe_zeilen_in_csv_ordner(ordnerpfad):
    gesamtanzahl_zeilen = 0

    # Überprüfen, ob der angegebene Pfad ein Verzeichnis ist
    if not os.path.isdir(ordnerpfad):
        print(f"{ordnerpfad} ist kein gültiges Verzeichnis.")
        return

    # Durchsuchen des Verzeichnisses nach CSV-Dateien
    for dateiname in os.listdir(ordnerpfad):
        if dateiname.endswith(".csv"):
            dateipfad = os.path.join(ordnerpfad, dateiname)

            # Lesen und Summieren der Zeilen in der CSV-Datei
            with open(dateipfad, 'r') as csv_datei:
                csv_reader = csv.reader(csv_datei)
                anzahl_zeilen = sum(1 for row in csv_reader)
                gesamtanzahl_zeilen += anzahl_zeilen

                print(f"Datei: {dateiname}, Zeilen: {anzahl_zeilen}")

    print(f"Gesamtanzahl der Zeilen in allen CSV-Dateien: {gesamtanzahl_zeilen}")




def run():
    # data_path = DATA_PATH  
    # cities = "GlobalLandTemperaturesByCity.csv"
    # if not os.path.exists(HISTOGRAMS):
    #         os.makedirs(HISTOGRAMS)
    
    # # analyse_same_val_cities() # TODO: runs tooooooooo long
    # # create_country_map(data_path,cities) # TODO: not working yet
    # create_latitude_histogram(data_path,cities)
    # create_longitute_histogram(data_path,cities)
    # create_continent_barchar(data_path,cities)
    # create_temperature_histogram(data_path,cities)
    # create_year_histogram(data_path,cities)
    # create_uncertainty_histogram(data_path,cities)
    # create_uncertainty_and_nans_per_year_barchar(data_path,cities)
    summe_zeilen_in_csv_ordner(DATA_PATH)
run()