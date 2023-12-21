import os
import re

folder_path = "models\models_lstm_rest\[0,0,0]1000.001[100,100,100]5025720280"
match = re.match(r'models\\models_lstm_rest\\(\[\d+(,\d+)*\])(\d+)(\.\d+)(\[\d+(,\d+)*\])(\d+)(\d+)(\d+)(\d+)', folder_path)

splits: list = folder_path.split("\\", 3)[-1]
print("splits: ", splits)
dropout: list = splits.split("]")[0][1:]
print("dropout", dropout)
batch_size: list = splits.split("]")[1][:3]
print("batch_size", batch_size)
architecture: list = splits.split("]")[1].split("[")[-1].split(",")
print("architecture", architecture)

# parameters = {
#             'dropout': dropout,
#             'batch_size': int(batch_size),
#             'learning_rate': float(learning_rate),
#             'architecture': architecture,
#             'steps_per_epoch': int(steps_per_epoch),
#             'epochs': int(epochs),
#             'seq_len_past': int(seq_len_past),
#             'seq_len_future': int(seq_len_future)
#         }


# if match:
#     dropout_list, batch_size, learning_rate, architecture_list, steps_per_epoch, epochs, seq_len_past, seq_len_future = match.groups()

#     # Convert the comma-separated strings to lists
#     dropout = [int(d) for d in dropout_list.strip('[]').split(',')]
#     architecture = [int(a) for a in architecture_list.strip('[]').split(',')]

#     parameters = {
#         'dropout': dropout,
#         'batch_size': int(batch_size),
#         'learning_rate': float(learning_rate),
#         'architecture': architecture,
#         'steps_per_epoch': int(steps_per_epoch),
#         'epochs': int(epochs),
#         'seq_len_past': int(seq_len_past),
#         'seq_len_future': int(seq_len_future)
#     }

# print("match: ", match)