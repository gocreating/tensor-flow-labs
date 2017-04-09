import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log1', type=str, help='Filename of log 1')
parser.add_argument('--log2', type=str, help='Filename of log 2')
args = parser.parse_args()

log1 = np.genfromtxt(args.log1, delimiter=',')
log2 = np.genfromtxt(args.log2, delimiter=',')

def plot_accuracy():
    line1, = plt.plot(
        log1[:, 0],
        log1[:, 1],
        color='blue',
        linewidth=1
    )
    line2, = plt.plot(
        log2[:, 0],
        log2[:, 1],
        color='orange',
        linewidth=1
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (100%)')
    plt.legend((line1, line2), ('retrain', 'train from random init'), loc='lower right')
    plt.savefig(
        './accuracy.jpg',
        dpi=400,
        format='jpg'
    )
    plt.clf()

def plot_error():
    line1, = plt.plot(
        log1[:, 0],
        1 - log1[:, 1],
        color='blue',
        linewidth=1
    )
    line2, = plt.plot(
        log2[:, 0],
        1 - log2[:, 1],
        color='orange',
        linewidth=1
    )
    plt.xlabel('Epoch')
    plt.ylabel('Test Error (100%)')
    plt.legend((line1, line2), ('retrain', 'train from random init'), loc='upper right')
    plt.savefig(
        './error.jpg',
        dpi=400,
        format='jpg'
    )
    plt.clf()

def plot_loss():
    line1, = plt.plot(
        log1[:, 0],
        log1[:, 2],
        color='blue',
        linewidth=1
    )
    line2, = plt.plot(
        log2[:, 0],
        log2[:, 2],
        color='orange',
        linewidth=1
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend((line1, line2), ('retrain', 'train from random init'), loc='upper right')
    plt.savefig(
        './loss.jpg',
        dpi=400,
        format='jpg'
    )
    plt.clf()

plot_accuracy()
plot_error()
plot_loss()
