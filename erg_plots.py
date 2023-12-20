import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.colors import LightSource


####
## ERGEBNISSE MLP -----------------------------------------------
###

batch_sizes = [25,25,25,25,50,50,50,50,100,100,100,100,200,200,200,200]
learning_rates = [0.001,0.0005,0.0001,0.00005,0.001,0.0005,0.0001,0.00005,0.001,0.0005,0.0001,0.00005,0.001,0.0005,0.0001,0.00005]

#1500,1500,1500,1500...
mae_first_arch = [1.1474,0.9882,0.8974,0.9026,0.9291,0.9187,0.8862,0.8845,0.8999,0.9186,0.8866,0.8758,0.8990,0.8983,0.8685,0.8599]
#1500,2000,5000, ...
mae_second_arch = [1.203,1.1705,0.9608,0.9592,1.09636,1.0891,0.9304,0.9074,1.0888,1.0998,0.9323,0.8758,1.0600,1.0388,0.8809,0.8759]
#2000#2000#2000
mae_3rd_arch = [2.6888,1.1561,0.9851,0.9173,1.0886,1.0827,0.8912,0.8964,1.0743,1.0743,0.8893,0.8774,1.0411,1.0269,0.8709,0.8640]


threshold1_mlp = 1.1
mae_first_arch_mlp_bounded = np.minimum(mae_first_arch, threshold1_mlp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_mlp_bounded,  linewidth=0.2, antialiased=True,color='darkblue')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE [1500, 1500, 1500, 1500, 1500] arch')

plt.show()
plt.close()


threshold2_mlp = 1.2
mae_second_arch_mlp_bounded = np.minimum(mae_second_arch, threshold2_mlp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_mlp_bounded,  linewidth=0.2, antialiased=True, color='darkred')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE [1500, 1500, 1500, 2000, 5000, 2000, 1000, 1000, 500] arch')

plt.show()
plt.close()

threshold3_mlp = 1.5
mae_third_arch_mlp_bounded = np.minimum(mae_3rd_arch, threshold3_mlp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_mlp_bounded,  linewidth=0.2, antialiased=True, color='green')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000] arch')

plt.show()
plt.close()

threshold3_mlp2 = 1.25
mae_third_arch_mlp_boundedall = np.minimum(mae_3rd_arch, threshold3_mlp2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch, linewidth=0.2, antialiased=True, color='darkblue', alpha=0.5, shade=False)
surface2 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch, linewidth=0.2, antialiased=True, color='darkred', alpha=0.5, shade=False)
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_mlp_boundedall, linewidth=0.2, antialiased=True, color='green',  alpha=0.5, shade=False)

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE arch compared')

plt.show()
plt.close()


####
## ERGEBNISSE LSTM -----------------------------------------------
###

batch_sizes = [25,25,25,25,25,50,50,50,50,50,75,75,75,75,75,100,100,100,100,100]
learning_rates = [0.001,0.01,0.005,0.05,0.03,  0.001,0.01,0.005,0.05,0.03,  0.001,0.01,0.005,0.05,0.03,  0.001,0.01,0.005,0.05,0.03]

#[10,10,10,10]
mae_first_arch_lstm = [2.9006,1.1846,1.1238,1.3592,1.1088,  1.169,1.1199,1.12,1.2164,1.1916,  1.0935,1.1103,1.1083,1.0306,4.3294,  1.1142,1.1147,1.135,1.1178,1.1387,7.2732]
#[20,20,20,20,20,20,20]
mae_second_arch_lstm = [4.2986,7.6412,5.4752,4.832,7.5445,  4.2545,5.6254,5.3813,7.8257,7.7649,  4.3270,5.9619,6.5134,7.883,6.3763,  4.3611,5.8823,7.8122,7.8342,5.8858]
#[30,30,30]
mae_third_arch_lstm = [4.2149,1.3196,1.1446,4.9263,3.4047,  4.2173,4.2846,1.0984,7.7712,4.2561,  1.1044,1.1931,4.2787,1.1247,1.1638,  1.0853,1.0374,1.043,7.9647,5.6361]
#[100,100,100,100]
mae_foth_arch_lstm = [1.0857,1.1795,1.0746,7.2851,7.6181,  1.0649,1.0678,1.047,7.758,5.8965,  1.087,1.1269,7.8030,4.4666,7.7531,  1.0708,1.1982,1.0939,4.4936,7.2732]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
threshold1_lstm = 1.4
mae_first_arch_lstm_bounded = np.minimum(mae_first_arch_lstm, threshold1_lstm)

surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_lstm_bounded, linewidth=0.2, antialiased=True, color='darkblue')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE [10,10,10,10] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_lstm, linewidth=0.2, antialiased=True, color='darkred')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE ')
ax.set_title('MAE [20,20,20,20,20,20,20] arch')

plt.show()
plt.close()
norm = Normalize()
colors = 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_lstm, linewidth=0.2, antialiased=True, color='green')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE [30,30,30,30,30] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_foth_arch_lstm, linewidth=0.2, antialiased=True, color='grey')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('MAE [100,100,100,100,100] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_lstm, linewidth=0.2, antialiased=True, color='darkblue',shade=False, alpha=0.5)
surface2 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_lstm, linewidth=0.2, antialiased=True, color='darkred',shade=False, alpha=0.5)
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_lstm, linewidth=0.2, antialiased=True, color='green',shade=False, alpha=0.55)
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_foth_arch_lstm, linewidth=0.2, antialiased=True, color='grey',shade=False, alpha=0.6)

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('compared arch')

plt.show()
plt.close()

####
## ERGEBNISSE CONV -----------------------------------------------
###

batch_sizes = [25,25,25,25, 50,50,50,50, 75,75,75,75, 100,100,100,100]
learning_rates = [0.0001,0.001,0.002,0.0004,  0.0001,0.001,0.002,0.0004,  0.0001,0.001,0.002,0.0004,  0.0001,0.001,0.002,0.0004]

#[10,10,10]
mae_first_arch_conv = [1.0389,0.9118,0.9544,0.8993, 1.0182,0.8836,0.9147,0.8774, 1.0136,0.8811,0.8877,0.8715, 0.9736,0.8756,0.8781,0.8602]
#[20,20,20,20,20,20,20]
mae_second_arch_conv = [1.0402,1.1118,1.0885,0.9075, 1.0025,1.0577,1.0867,0.8700, 0.9995,1.0551,1.0766,0.8648, 0.9630,1.0439,1.1034,0.8717]
#[50,50,50,50,50]
mae_third_arch_conv = [1.0050,1.1,1.0945,0.9224, 0.9911,0.9176,1.0933,0.8958, 0.9785,0.9371,1.1042,0.8650, 0.9478,0.8899,1.0912,0.858]
#[50,50,50]
mae_foth_arch_conv = [1.0045,0.9416,1.0549,0.9204, 0.9818,0.8896,0.9527,0.8630, 0.9937,0.8632,0.8952,0.8644, 0.9920,0.8691,1.0586,0.8614]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_conv, linewidth=0.2, antialiased=True, color='darkblue')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE ')
ax.set_title('MAE [[10, 10, 10], [5, 5, 5]] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_conv, linewidth=0.2, antialiased=True, color='darkred')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE ')
ax.set_title('MAE [[20, 20, 20, 20, 20], [30, 30, 30, 30, 30]] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_conv, linewidth=0.2, antialiased=True, color='green')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE ')
ax.set_title('MAE [[50, 50, 50, 50, 50], [10, 20, 20, 20, 10]] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_foth_arch_conv, linewidth=0.2, antialiased=True, color='grey')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE ')
ax.set_title('MAE [[50, 50, 50], [10, 10, 10]] arch')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_conv, linewidth=0.2, antialiased=True, color='darkblue',shade=False, alpha=0.5)
surface2 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_conv, linewidth=0.2, antialiased=True, color='darkred',shade=False, alpha=0.5)
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_conv, linewidth=0.2, antialiased=True, color='green',shade=False, alpha=0.55)
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_foth_arch_conv, linewidth=0.2, antialiased=True, color='grey',shade=False, alpha=0.6)

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')
ax.set_title('compared arch')

plt.show()
plt.close()