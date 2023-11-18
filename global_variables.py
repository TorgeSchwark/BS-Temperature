VALIDATION_PERCENTAGE = 0.5
DATA_PATH = 'data_To_Process\\'
DATA_PATH_PROCESSED = 'processed_Data\\'
HISTOGRAMS = 'histograms\\'

CITYS ='GlobalLandTemperaturesByCity.csv'
COUNTRYS = 'GlobalLandTemperaturesByCountry.csv'
MAJORCITYS = 'GlobalLandTemperaturesByMajorCity.csv'
STATE = 'GlobalLandTemperaturesByState.csv'

GPU_STRING = '/gpu:0'
BATCH_SIZE = 100
MODEL_NAME = "Tests"
EPOCHS = 100
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 32
SEQ_LEN_PAST = 1000
SEQ_LEN_FUTURE = 360
NUM_INPUT_PARAMETERS = 1
NUM_OUTPUT_PARAMETERS = 1
VALIDATION_PERCENTAGE = 0.3
MAX_LOSS = 30