VALIDATION_PERCENTAGE = 0.5
DATA_PATH = 'data_To_Process\\'
DATA_PATH_PROCESSED = 'processed_Data\\'
HISTOGRAMS = 'histograms\\'
UMFRAGE_PATH = 'human_prediction\\'
DUPLICATES = 'duplicates\\'

CITYS ='GlobalLandTemperaturesByCity.csv'
COUNTRYS = 'GlobalLandTemperaturesByCountry.csv'
MAJORCITYS = 'GlobalLandTemperaturesByMajorCity.csv'
STATE = 'GlobalLandTemperaturesByState.csv'

GPU_STRING = '/gpu:0'
BATCH_SIZE = 100
MODEL_NAME = "Mlp"
EPOCHS = 50
STEPS_PER_EPOCH = 50 # 30
VALIDATION_STEPS = 32
# in months
SEQ_LEN_PAST = 840 # 70 years       
# TODO: "Wie verhält sich die Performance einer beliebigen Architektur, wenn die Länge der Input-Sequenzen von 8, 16, 32, 64 verändert wird?"
# SEQ_LEN_PAST = 8
# SEQ_LEN_PAST = 16
# SEQ_LEN_PAST = 32
# SEQ_LEN_PAST = 64

SEQ_LEN_FUTURE = 300 # 25 years prediction
NUM_INPUT_PARAMETERS = 1
NUM_OUTPUT_PARAMETERS = 1
# percentage of data used for validation
VALIDATION_PERCENTAGE = 0.3
# only show if error is lower this value
MAX_LOSS = 50 # 30