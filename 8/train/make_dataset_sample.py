import pandas
import math
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_finance

SAVE_IMG_DIR = "img"


def main():
    df = pandas.read_csv("csv/sample.csv")
    save_candlestick_img(df, "sample.jpg")
    save_candlestick_with_volume_img(df, "sample2.jpg")
        
def save_candlestick_img(df, save_filepath):
    open_value = numpy.array(df["open"])
    low_value = numpy.array(df["low"])
    high_value = numpy.array(df["high"])
    close_value = numpy.array(df["close"])
    x_list = [i for i in range(len(df))]
    fig = plt.figure(figsize=(2.5, 2.5), facecolor="k", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0)
    ax.patch.set_facecolor('black')
    ax.patch.set_alpha(1)
    mpl_finance.candlestick2_ohlc(ax, opens=open_value,
                                  highs=high_value,
                                  lows=low_value,
                                  closes=close_value,
                                  width=1, alpha=1, colorup='r', colordown='g')
    
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    ax = plt.gca() 
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.style.context('classic')
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.savefig(save_filepath)
    plt.close("all")

def save_candlestick_with_volume_img(df, save_filepath):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True,
                         gridspec_kw={'height_ratios': [4, 1]})
    mpl_finance.candlestick2_ochl(ax[0], df['open'], df['close'], df['high'], df['low'],
                                  width=1, colorup='r', colordown='g')
    ax[0].grid(False)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[0].axis('off')
    mpl_finance.volume_overlay(ax[1], df['open'], df['close'], df['volume'],
                            colorup='r', colordown='g', width=1)
    ax[1].grid(False)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[1].axis('off')
    plt.savefig(save_filepath)
    plt.close("all")

if __name__ == "__main__":
    main()