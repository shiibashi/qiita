import pandas
import numpy
import os
import random
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_finance
import gc

DIR_PATH = os.path.dirname(__file__)
os.chdir("{}/..".format(DIR_PATH)) # script/ に移動

def select_chart_data(df):
    ymd = df.sample(1)["Date"].values[0]
    ymd_dt = datetime.datetime.strptime(ymd, "%Y-%m-%d")
    ymd_dt_6m = ymd_dt - datetime.timedelta(days=180)
    ymd_6m = ymd_dt_6m.strftime("%Y-%m-%d")
    chart_data = df.query("Date >= @ymd_6m and Date <= @ymd").reset_index(drop=True)
    return chart_data

def _filename(chart_data):
    code = chart_data["Code"].values[0]
    from_ymd = chart_data["Date"].values[0]
    to_ymd = chart_data["Date"].values[-1]
    filename = "{}_{}_{}.png".format(code, from_ymd, to_ymd)
    return filename


def save_candlestick_img_with_volume(df, save_filepath):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(2.56, 2.56), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
    mpl_finance.candlestick2_ochl(ax[0], df['Open'], df['Close'], df['High'], df['Low'],
                                  width=1, colorup='r', colordown='g')
    x_list = [i for i in range(len(df))]
    ma5_value = numpy.array(df["Open_MA_5"])
    ma25_value = numpy.array(df["Open_MA_25"])
    ma75_value = numpy.array(df["Open_MA_75"])
    
    ax[0].plot(x_list, ma5_value, markersize=2, color='black')
    ax[0].plot(x_list, ma25_value, markersize=2, color='y')
    ax[0].plot(x_list, ma75_value, markersize=2, color='b')
    ax[0].grid(False)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    y_min = min(df["Low"].min(), ma5_value.min(), ma25_value.min(), ma75_value.min())
    y_max = max(df["High"].max(), ma5_value.max(), ma25_value.max(), ma75_value.max())
    ax[0].set_ylim([y_min, y_max])
    ax[0].axis('off')
    mpl_finance.volume_overlay(ax[1], df['Open'], df['Close'], df['volume'],
                            colorup='black', colordown='black', width=2)
    ax[1].grid(False)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[1].axis('off')
    plt.savefig(save_filepath)
    plt.close("all")
    gc.collect()

def save_candlestick_img_only_ma(df, save_filepath):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(0.64, 0.64), sharex=True)
    x_list = [i for i in range(len(df))]
    ma5_value = numpy.array(df["Open_MA_5"])
    ma25_value = numpy.array(df["Open_MA_25"])
    ma75_value = numpy.array(df["Open_MA_75"])

    ax.plot(x_list, ma5_value, markersize=2, color='black')
    ax.plot(x_list, ma25_value, markersize=2, color='y')
    ax.plot(x_list, ma75_value, markersize=2, color='b')
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    y_min = min(df["Low"].min(), ma5_value.min(), ma25_value.min(), ma75_value.min())
    y_max = max(df["High"].max(), ma5_value.max(), ma25_value.max(), ma75_value.max())
    ax.set_ylim([y_min, y_max])
    ax.axis('off')
    plt.savefig(save_filepath)
    plt.close("all")
    gc.collect()


def run(n, with_volume=False):
    filename_list = os.listdir("data/code_data")
    os.makedirs("data/img/clustering", exist_ok=True)
    for i in range(n):
        if i % 1000 == 0:
            print(i, flush=True)
        filename = random.sample(filename_list, 1)[0]
        df = pandas.read_csv("data/code_data/{}".format(filename)).reset_index(drop=True)
        if len(df) <= 100:
            continue
        chart_data = select_chart_data(df)
        if len(chart_data) <= 90 or chart_data["Volume"].mean() < 20000:
            continue
        save_filepath = "data/img/clustering/{}".format(_filename(chart_data))
        if with_volume:
            save_candlestick_img_with_volume(chart_data, save_filepath)
        else:
            save_candlestick_img_only_ma(chart_data, save_filepath)

def select_train_test(test_num=20):
    filelist = os.listdir("data/img/clustering")
    sample_list = random.sample(filelist, test_num)
    os.makedirs("data/img/test/ccae_test", exist_ok=True)
    os.makedirs("data/img/train/ccae_train", exist_ok=True)
    for filename in filelist:
        if filename in sample_list:
            os.system("cp data/img/clustering/{} data/img/test/ccae_test/{}".format(filename, filename))
        else:
            os.system("cp data/img/clustering/{} data/img/train/ccae_train/{}".format(filename, filename))
