import matplotlib.pyplot as plt
import pandas as pd


def create_longitute_histogram(data_path,cities):
    df = pd.read_csv(data_path+cities)
    df['Longitude'] = df['Longitude'].str.extract('(\d+\.\d+)').astype(float)*(-1+ (df['Longitude'].str.extract('([A-Z]+)')=="E")*2)

    rounded_longitude = (df['Longitude'] // 10) * 10

    print(rounded_longitude[:5])
    print(len(rounded_longitude))

    plt.hist(rounded_longitude, bins=36, color='blue', edgecolor='black')

    plt.title('Histogramm der Längengrade')
    plt.xlabel('Längengrade')
    plt.ylabel('Anzahl')

    plt.xticks(rotation=45, ha='right')

    plt.show()

def run():
    data_path = "C:\\Users\\Torge\\Desktop\\Uni\\5Semester\\Bachelor Seminar\\Datensatz_Erderwärmung\\dataToProcess\\"    
    cities = "GlobalLandTemperaturesByCity.csv"
    
    create_longitute_histogram(data_path,cities)

run()