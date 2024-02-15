import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from reservoir import Wrec, Wi, inp

sim_id = 1
eta = 1
frequency = 1
A = 1

# weights are saved in ...
sub_folder = "/eta_" + str(eta) + "_frequency_" + str(frequency) + "_amplitude_" + str(A) + "/run_" + str(sim_id) + "/"
folder_net = "./results" + sub_folder
# plots will be saved in ...
plot_folder = "plots" + sub_folder
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

weights = {
    Wi.name: np.load(folder_net + "w_" + Wi.name + ".npy"),
    Wrec.name: np.load(folder_net + "w_" + Wrec.name + ".npy")
}

# reservoir weights
fig1, axs = plt.subplots(ncols=3, figsize=(10, 5))
axs[0].imshow(weights[Wrec.name][0, :, :], cmap='RdBu', vmin=-0.1, vmax=0.1)
axs[0].set_title("Init")
axs[1].imshow(weights[Wrec.name][1, :, :], cmap='RdBu', vmin=-0.1, vmax=0.1)
axs[1].set_title("After Learning")
axs[2].imshow(weights[Wrec.name][1, :, :] - weights['w_lat_res'][0, :, :], cmap='RdBu', vmin=-0.05, vmax=0.05)
axs[2].set_title("Weight changes")

plt.savefig(plot_folder + f"weight_changes_sim{sim_id}.pdf")
plt.close(fig1)

# input weights
fig2 = plt.figure(figsize=(8, 10))

n_parameters = 9

for parameter in range(n_parameters):
    ax = plt.subplot(3, 3, 1 + parameter)
    ax.imshow(weights[Wi.name][0, :, parameter].reshape(20, 20), cmap='RdBu', vmin=-0.2, vmax=0.2)

plt.savefig(plot_folder + f"input_weights_sim{sim_id}.pdf")
plt.close(fig2)
