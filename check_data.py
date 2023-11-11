import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

HISTOGRAM_PATH = "C:\\Users\\Torge\\Desktop\\Uni\\5Semester\\Bachelor Seminar\\Datensatz_Erderwärmung\\Histograms\\"
LONGITUDE = "Longitude"
LATITUDE = "Latitude"
CONTINENTS = "Continents"
TEMPERATURE = "Temperatur"
YEAR = "Years"
UNCERTAINTY = "Uncertainty"
UNCERTAINTY_PER_YEAR = "Uncertainty_per_year"

##TODO uncertainty per year and inconsistant data per year

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

def create_longitute_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
    df['Longitude'] = df['Longitude'].str.extract('(\d+\.\d+)').astype(float)*(-1+ (df['Longitude'].str.extract('([A-Z]+)')=="E")*2)

    plt.hist(df['Longitude'], bins=36, color='blue', edgecolor='black')

    plt.title('Histogramm of Longitude')
    plt.xlabel('Longitude')
    plt.ylabel('Anzahl')
    plt.tight_layout()

    
    if HISTOGRAM_PATH+LONGITUDE:
        plt.savefig(HISTOGRAM_PATH+LONGITUDE)  

    plt.show()

def create_latitude_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
    df['Latitude'] = df['Latitude'].str.extract('(\d+\.\d+)').astype(float)*(-1+ (df['Latitude'].str.extract('([A-Z]+)')=="N")*2)

    plt.hist(df['Latitude'], bins=18, color='blue',edgecolor='black')

    plt.title('Histogramm of Latitude')
    plt.xlabel('Latitude')
    plt.ylabel('Anzahl')
    plt.tight_layout()
    
    if HISTOGRAM_PATH+LATITUDE:
        plt.savefig(HISTOGRAM_PATH+LATITUDE)  

    plt.show()

def create_continent_histogram(data_path,cities):

    df = pd.read_csv(data_path+cities)
    country_dict = {}
    continent_dict = {}
    #initation of Countrys
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
    plt.title('Population Distribution by Continent')
    plt.xlabel('Continent')
    plt.xticks(rotation=45)
    plt.ylabel('Population')
    plt.tight_layout()

    if HISTOGRAM_PATH+CONTINENTS:
        plt.savefig(HISTOGRAM_PATH+CONTINENTS)  

    plt.show()

def create_temperature_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
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

    if HISTOGRAM_PATH+TEMPERATURE:
        plt.savefig(HISTOGRAM_PATH+TEMPERATURE)  
    
    plt.show()

def create_year_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)

    df['dt'] = pd.to_datetime(df['dt']).dt.year

    plt.hist(df['dt'], bins=271, color='blue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Histogramm of Temperatur')
    plt.xlabel('Temperatur')
    plt.ylabel('Anzahl')
    plt.tight_layout()

    if HISTOGRAM_PATH+YEAR:
        plt.savefig(HISTOGRAM_PATH+YEAR)  
    
    plt.show()

def create_uncertainty_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)

    plt.hist(df['AverageTemperatureUncertainty'], bins=100, color='blue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Histogramm of Temperatur')
    plt.xlabel('Temperatur')
    plt.ylabel('Anzahl')
    plt.tight_layout()

    if HISTOGRAM_PATH+UNCERTAINTY:
        plt.savefig(HISTOGRAM_PATH+UNCERTAINTY)  
    
    plt.show()

def run():
    data_path = "C:\\Users\\Torge\\Desktop\\Uni\\5Semester\\Bachelor Seminar\\Datensatz_Erderwärmung\\dataToProcess\\"    
    cities = "GlobalLandTemperaturesByCity.csv"
    
    # create_latitude_histogram(data_path,cities)
    # create_longitute_histogram(data_path,cities)
    # create_continent_histogram(data_path,cities)
    # create_temperature_histogram(data_path,cities)
    # create_year_histogram(data_path,cities)
    create_uncertainty_histogram(data_path,cities)

run()