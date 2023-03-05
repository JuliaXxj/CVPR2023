import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import sys
import argparse

# args = sys.argv
# data_name = args[1]     # 'svhn', 'cifar10', 'cifar100'
# model_name = args[2]    # 'resnet34', 'resnet18', 'vgg16', 'vgg13', 'vgg11'


parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model', type=str,
                    choices=['resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11', 'nf_resnet18', 'nf_resnet34'],
                    required=True, help="choose model")
parser.add_argument('-d', '--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100'], required=True,
                    help="choose dataset")
parser.add_argument('-b', '--bias', type=bool, default=False, help="Model with bias or without bias")

args = parser.parse_args()

data_name = args.dataset
model_name = args.model
bias = args.bias

runs = 'runs'
if bias:
    log_dir = os.path.join(runs, data_name, model_name, 'bias', 'log')
    plot_path = os.path.join(runs, data_name, model_name, 'bias', "energy_plot.png")
else:
    log_dir = os.path.join(runs, data_name, model_name, 'nobias', 'log')
    plot_path = os.path.join(runs, data_name, model_name, 'nobias', "energy_plot.png")



file_path = os.path.join(log_dir, 'train energy', 'train_energy.csv')
test_path = os.path.join(log_dir, 'test_acc.csv')
test_results = pd.read_csv(test_path)
test_epoch = test_results['steps'].astype(int)
test_error = 1 - test_results['values']

noise_path = os.path.join(log_dir, 'noise_acc.csv')
noise_results = pd.read_csv(noise_path)
noise_epoch = noise_results['steps'].astype(int)
noise_error = 1 - noise_results['values']


data = pd.read_csv(file_path).values
fft_epoch = data[0:, 0].astype(int)
energy = data[0:, 1:].transpose(1, 0)
delta_energy = np.log(energy / np.sum(energy, axis=0, keepdims=True))

fig = plt.figure(figsize=(5, 3))

ax = fig.add_subplot(111)

cmap = sb.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True)
heatmap = ax.contourf(fft_epoch[:], range(delta_energy.shape[0]), delta_energy, 30, cmap=cmap)
ax.set_ylabel('frequency', fontsize=14)
ax.set_xlabel('epoch', fontsize=14)
ax.set_xlim([min(fft_epoch), max(fft_epoch)])

ax2 = ax.twinx()

ax2.plot(test_epoch, test_error, color='darkorange', label='test')
ax2.plot(noise_epoch, noise_error, color='red', label='perturbed')
ax2.set_ylabel('error', fontsize=14)

fig.colorbar(heatmap, pad=0.15, ax=ax2)

plt.grid(False)
plt.xscale('log')

plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
