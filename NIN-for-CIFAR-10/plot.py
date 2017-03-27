import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, help='Filename of logs')
args = parser.parse_args()

filename = args.log or './log.csv'
logs = np.genfromtxt(filename, delimiter=',')

def plot_accuracy():
    line_accuracy = plt.plot(
        logs[:, 0],
        logs[:, 1],
        color='blue',
        linewidth=1
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(100%)')
    plt.savefig(
        './accuracy.jpg',
        dpi=400,
        format='jpg'
    )
    plt.clf()

def plot_loss():
    line_loss = plt.plot(
        logs[:, 0],
        logs[:, 2],
        color='green',
        linewidth=1
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(
        './loss.jpg',
        dpi=400,
        format='jpg'
    )
    plt.clf()

plot_accuracy()
plot_loss()
