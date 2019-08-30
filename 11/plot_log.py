import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy
import gc

def plot(logger):
    _plot(logger.train_log, "train.png")
    _plot(logger.test_log, "test.png")
    _plot(logger.test2_log, "test2.png")

def _plot(log, filepath):
    n = len(log)
    v = [t[-1] for t in log]
    x = numpy.array(range(n))
    cutline = x * 0 + 0.9
    plt.scatter(x, v, c="r")
    plt.plot(x, cutline, c="b")
    plt.savefig(filepath)
    plt.close("all")
    gc.collect()

def _plot_all(log, filepath):
    n = len(log)
    v = [t[-1] for t in log]
    x = numpy.array(range(n))
    cutline = x * 0 + 0.9
    plt.scatter(x, v, c="r")
    plt.plot(x, cutline, c="b")
    plt.savefig(filepath)
    plt.close("all")
    gc.collect()
