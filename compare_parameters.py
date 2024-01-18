from tensorflow.keras.models import load_model
import os

def print_model_parameters(model_path):
    # Load the model
    try:
        model = load_model(model_path)

        # Get the number of parameters
        params = model.count_params()

        # Print the number of parameters
        print(f'The model has {params} parameters.')
    except:
        print('Error loading model: ', model_path)

# Example usage:
# print_model_parameters('path_to_your_model.h5')
    
# loop over all 4 architectures and print the number of parameters
path  = 'models/best_models/'
if not os.path.exists(path):
    os.makedirs(path)
    print('Created directory: ', path)

model_names = ['model_1.h5', 'model-042-mae1.2158.h5', 'model_3.h5', 'model_4.h5'] # TODO: Namen eintragen
for model_n in model_names:
    model_path = path + model_n
    print(f'Num of parameters in model: {model_n}:')
    print_model_parameters(model_path)
    # print('\n')