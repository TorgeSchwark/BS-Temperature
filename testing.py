import math

def parse_file(file_name):
    file = open(file_name,"r")
    lines = file.readlines()

    samples = []
    count = 0
    for line in lines[0:100]:
        x, y = line.strip().split(' ')
        print(math.isnan(float(y)))

    return samples

parse_file("processed_data\\Zambia.txt")