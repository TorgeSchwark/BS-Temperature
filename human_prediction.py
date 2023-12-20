from global_variables import * 
from data_parser import *
import os
import random
import os
import matplotlib.pyplot as plt

UMFRAGE_MENGE = 30
#Plotts data und combines graphs to Pdfs so predictions can be made more easily
def make_prediction_graphs():
  Umfrage_menge = UMFRAGE_MENGE

  if not os.path.exists("UMFRAGE_PATH"):
    os.makedirs("UMFRAGE_PATH")
  
  data, dict = get_all_files_as_list()

  for i in range(Umfrage_menge):

    data_ind = random.randint(0,len(data)-1)
    ind_in_data = random.randint(0,len(dict[data_ind])-1)

    data_to_plot = data[data_ind][dict[data_ind][ind_in_data]-SEQ_LEN_FUTURE-SEQ_LEN_PAST:dict[data_ind][ind_in_data]-SEQ_LEN_FUTURE]
    labels = data[data_ind][dict[data_ind][ind_in_data]-SEQ_LEN_FUTURE:dict[data_ind][ind_in_data]]
    
    plot = plt.figure()
    plt.plot(data_to_plot)
    plt.title('Temperature')
    plt.xlabel('Monate')
    plt.ylabel('Temperatur')

    if not os.path.exists(UMFRAGE_PATH+ str(i) + "\\"):
      os.makedirs(UMFRAGE_PATH+ str(i) + "\\")

    x_ticks = np.arange(0, len(data_to_plot) + SEQ_LEN_FUTURE, step=1)
    plt.xticks(x_ticks, fontsize='xx-small', rotation=90)

    plt.grid(True, which='both', linestyle='-', linewidth=1)

    plt.savefig(UMFRAGE_PATH + str(i) + "\\" + 'plot.pdf',format='pdf', dpi=600)

    with open(UMFRAGE_PATH+ str(i) + "\\" + "labels.txt" , 'w') as file:
      for item in labels:
        file.write(str(item)+ '\n')
    
    with open(UMFRAGE_PATH+ str(i) + "\\" + "Predictions.txt" , 'w') as file:
      for count in range(len(labels)):
        file.write("prediction"+str(count)+": " '\n')
    plt.close()
  
def show_human_prediction():
  data, dict = get_all_files_as_list()
  cleane_data = []
  cleane_file = []

  for file in data:
    cleane_file = []
    for value in file:
      cleane_file.append(value[0])
    cleane_data.append(cleane_file)
  
  label_array = []
  global_mse = 0
  global_mae = 0
  for i in range(UMFRAGE_MENGE):
    # found = False
    # label_array = []
    # with open(UMFRAGE_PATH+ str(i) +"\\" + "labels.txt") as datei:
    #   for zeile in datei:
    #     wert = float(zeile.strip("()\n,"))
    #     label_array.append(wert)

    # prediction_array = []
    # with open(UMFRAGE_PATH+ str(i) +"\\" + "Predictions.txt") as datei:
    #   for zeile in datei:
    #     wert = float(zeile.strip("\n").split()[1])
    #     prediction_array.append(wert)
    
    # #find data and plot it
    # for s in range(len(cleane_data)): 
    #   for m in range(len(cleane_data[s])):

    #     if len(cleane_data[s]) > m+len(label_array):
    #       if all(cleane_data[s][m+k] == label_array[k] for k in range(len(label_array))) and m-SEQ_LEN_PAST >= 0 and not found: 
    #         found = True
    #         plt.figure()
    #         plt.plot(list(range(SEQ_LEN_PAST+SEQ_LEN_FUTURE)),cleane_data[s][m-SEQ_LEN_PAST:m+SEQ_LEN_FUTURE])
    #         plt.plot(list(range(SEQ_LEN_PAST,SEQ_LEN_PAST+SEQ_LEN_FUTURE)),prediction_array)
            
            
    #         plt.title('Temperature')
    #         plt.xlabel('Monate')
    #         plt.ylabel('Temperatur')
    #         plt.xticks(list(range(0,SEQ_LEN_PAST+SEQ_LEN_FUTURE)), fontsize='xx-small', rotation=90)

    #         plt.grid(True, which='both', linestyle='-', linewidth=1)

    #         plt.savefig(UMFRAGE_PATH + str(i) + "\\" + str(s) + str(m) + 'plot.pdf',format='pdf', dpi=600)
    #         plt.close()

    #culculate mae and mse
    mae = 0
    mse = 0
    with open(UMFRAGE_PATH+ str(i) + "\\" + "labels.txt") as labels:
      with open(UMFRAGE_PATH+ str(i) + "\\" + "Predictions.txt") as predictions:
        for zeile_nummer, (label, prediction) in enumerate(zip(labels, predictions), start=1):
          pred = float(prediction.strip("\n").split()[1])
          label = float(label.strip('()\n,'))
          mae += abs(pred-label)
          mse += abs(pred-label)*abs(pred-label)
    mae = mae/zeile_nummer
    mse = mse/zeile_nummer
    global_mse += mse
    global_mae += mae

    with open(UMFRAGE_PATH+ str(i) + "\\" + "mse_mae.txt", 'w') as datei:
      datei.write(f'MSE: {mse}\nMAE: {mae}')

  global_mse = global_mse/UMFRAGE_MENGE
  global_mae = global_mae/UMFRAGE_MENGE
  with open(UMFRAGE_PATH + "global_mse_mae.txt", 'w') as datei:
      datei.write(f'MSE: {global_mse}\nMAE: {global_mae}')       

show_human_prediction()