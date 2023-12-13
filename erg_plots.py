import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# [25,50,100,200]
# [0.001,0.0005,0.0001,0.00005]

batch_sizes = [25,25,25,25,50,50,50,50,100,100,100,100,200,200,200,200]
learning_rates = [0.001,0.0005,0.0001,0.00005,0.001,0.0005,0.0001,0.00005,0.001,0.0005,0.0001,0.00005,0.001,0.0005,0.0001,0.00005]

#1500,1500,1500,1500...
mae_first_arch = [1.1474,0.9882,0.8974,0.9026,0.9291,0.9187,0.8862,0.8845,0.8999,0.9186,0.8866,0.8758,0.8990,0.8983,0.8685,0.8599]
#1500,2000,5000, ...
mae_second_arch = [1.203,1.1705,0.9608,0.9592,1.09636,1.0891,0.9304,0.9074,1.0888,1.0998,0.9323,0.8758,1.0600,1.0388,0.8809,0.8759]
#2000#2000#2000
mae_3rd_arch = [2.6888,1.1561,0.9851,0.9173,1.0886,1.0827,0.8912,0.8964,1.0743,1.0743,0.8893,0.8774,1.0411,1.0269,0.8709,0.8640]



# 3D-Oberfläche
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_3rd_arch,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch,  linewidth=0.2, antialiased=True, cmap='viridis')
surface2 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch,  linewidth=0.2, antialiased=True, cmap='plasma')
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_3rd_arch,  linewidth=0.2, antialiased=True, cmap='inferno')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

batch_sizes = [25,25,25,25,50,50,50,50,75,75,75,75,100,100,100,100]
learning_rates = [0.001,0.01,0.005,0.05,0.001,0.01,0.005,0.05,0.001,0.01,0.005,0.05,0.001,0.01,0.005,0.05]

#[10,10,10,10]
mae_first_arch_lstm = [2.9006,1.1846,1.1238,1.3592,1.169,1.1199,1.12,1.2164,1.0935,1.1103,1.1083,1.0306,1.1142,1.1147,1.135,1.1178]
#[20,20,20,20,20,20,20]
mae_second_arch_lstm = [4.2986,7.6412,5.4752,4.832,4.2545,5.6254,5.3813,7.8257,4.3270,5.9619,6.5134,7.883,4.3611,5.8823,7.8122,7.8342]
#[30,30,30]
mae_third_arch_lstm = [4.2149,1.3196,1.1446,4.9263,4.2173,4.2846,1.0984,7.7712,1.1044,1.1931,4.2787,1.1247,1.0853,1.0374,1.043,7.9647]
#[100,100,100,100]
mae_foth_arch_lstm = [1.0857,1.1795,1.0746,7.2851,1.0649,1.0678,1.047,7.758,1.087,1.1269,7.8030,4.4666,1.0708,1.1982,1.0939,4.4936]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_lstm,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

print(mae_third_arch_lstm)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_lstm,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_lstm,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_foth_arch_lstm,  linewidth=0.2, antialiased=True, cmap='cividis_r')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface1 = ax.plot_trisurf(batch_sizes, learning_rates, mae_first_arch_lstm,  linewidth=0.2, antialiased=True, cmap='viridis')
surface2 = ax.plot_trisurf(batch_sizes, learning_rates, mae_second_arch_lstm,  linewidth=0.2, antialiased=True, cmap='plasma')
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_third_arch_lstm,  linewidth=0.2, antialiased=True, cmap='inferno')
surface3 = ax.plot_trisurf(batch_sizes, learning_rates, mae_foth_arch_lstm,  linewidth=0.2, antialiased=True, cmap='magma')

ax.set_xlabel('Batchsize')
ax.set_ylabel('Learning rate')
ax.set_zlabel('MAE')

plt.show()
plt.close()
