from global_variables import * 
from data_parser import *
import os
import random
import os
from PyPDF2 import PdfFileReader, PdfFileWriter
import matplotlib.pyplot as plt


#Plotts data und combines graphs to Pdfs so predictions can be made more easily
def run():
  Umfrage_menge = 60

  if not os.path.exists(UMFRAGE_PATH):
    os.makedirs(UMFRAGE_PATH)
  
  data, dict = get_all_files_as_list(DATA_PATH_PROCESSED)

  for i in range(Umfrage_menge):

    data_ind = random.randint(0,len(data)-1)
    ind_in_data = random.randint(0,len(dict[data_ind])-1)

    data_to_plot = data[data_ind][dict[data_ind][ind_in_data]-SEQ_LEN_FUTURE-SEQ_LEN_PAST:dict[data_ind][ind_in_data]-SEQ_LEN_FUTURE]
    
    plot = plt.figure()
    plt.plot(data_to_plot)
    plt.title('Temperature')
    plt.xlabel('Monate')
    plt.ylabel('Temperatur')

    x_ticks = np.arange(0, len(data_to_plot) + SEQ_LEN_FUTURE, step=1)
    plt.xticks(x_ticks, fontsize='xx-small', rotation=90)

    plt.grid(True, which='both', linestyle='-', linewidth=1)

    plt.savefig(UMFRAGE_PATH + str(i) + 'plot.pdf',format='pdf', dpi=600)
    
    plt.close()

  combine_pdfs(UMFRAGE_PATH, UMFRAGE_PATH + "together.pdf")



run()