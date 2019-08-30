import pandas
import numpy
import matplotlib.pyplot as plt
import datetime
import calendar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_finance
import gc
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class PipeLine(object):
    def __init__(self):
        normalize_params = None

    def run(self, df):
        df = self.append_feature(df)
        df = self.normalize(df)
        df = df.dropna().reset_index(drop=True)
        return df
        
    def append_feature(self, df):
        df = df.copy()
        df["Open_MA_5"] = df["Open"].rolling(5).mean()
        df["Open_MA_25"] = df["Open"].rolling(25).mean()
        df["Open_MA_75"] = df["Open"].rolling(75).mean()
        df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
        df["Profit"] = df["Open"].pct_change(1).shift(-1)
        return df

    def normalize(self, df):
        df = df.copy()
        df["Open_MA_5_N"] = df["Open_MA_5"] / df["Open"]
        df["Open_MA_25_N"] = df["Open_MA_25"] / df["Open"]
        df["Open_MA_75_N"] = df["Open_MA_75"] / df["Open"]
        df["Volume_MA_5_N"] = df["Volume_MA_5"] / df["Volume"]
        df["Open_R_1"] = df["Open"] / df["Open"].shift(1)
        return df

def save_candlestick_img_with_volume(df, save_filepath):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(2.5, 2.5), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
    mpl_finance.candlestick2_ochl(ax[0], df['Open'], df['Close'], df['High'], df['Low'],
                                  width=2, colorup='r', colordown='g')
    x_list = [i for i in range(len(df))]
    ma5_value = numpy.array(df["Open_MA_5"])
    ma25_value = numpy.array(df["Open_MA_25"])
    ma75_value = numpy.array(df["Open_MA_75"])
    ax[0].plot(x_list, ma5_value, markersize=3, color='m')
    ax[0].plot(x_list, ma25_value, markersize=3, color='y')
    ax[0].plot(x_list, ma75_value, markersize=3, color='b')
    ax[0].grid(False)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[0].axis('off')
    mpl_finance.volume_overlay(ax[1], df['Open'], df['Close'], df['Volume'],
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


def _append_feature(df):
    df = df.copy()
    df["Date"] = df["Year"].apply(str) +"-"+ df["Month"].apply(lambda x: str(x).zfill(2)) +"-"+ df["Day"].apply(lambda x: str(x).zfill(2))

    df["Open_MA_5"] = df["Open"].rolling(5).mean()
    df["Open_MA_25"] = df["Open"].rolling(25).mean()
    df["Open_MA_75"] = df["Open"].rolling(75).mean()
    df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
    
    return df

if __name__ == "__main__":
    df = pandas.read_csv("dataset/btc.csv")
    n = len(df)
    df = _append_feature(df)

    pipeline = PipeLine()
    feature = pipeline.run(df)
    feature.to_csv("dataset/feature.csv", index=False)
    for col in ["Open_MA_5_N", "Open_MA_25_N", "Open_MA_75_N", "Open_R_1"]:
        s = feature[col]
        feature[col] = (s - s.min()) / (s - s.min()).max()
    feature.to_csv("dataset/feature_normalized.csv", index=False)

    try:
        os.mkdir("dataset/img")
    except FileExistsError:
        pass

    file_list = os.listdir("dataset/img")
    import time
    def _task(i):
        if i % 1000 == 0:
            print(i, flush=True)
        chart_data = feature[i:i+24*7].reset_index(drop=True)
        ts_2 = chart_data["Timestamp"].values[-1]
        filename = "{}.png".format(ts_2)
        #print(feature[i:i+24*5])
        #chart_data.to_csv("a.csv")
        #assert False
        if filename not in file_list:
            save_candlestick_img_with_volume(chart_data, "dataset/img/{}".format(filename))

    #print("iteration: {}".format(len(feature)-24*5), flush=True)
    with ProcessPoolExecutor(max_workers=3) as executor:
        for i in range(len(feature)-24*7):
            executor.submit(_task, i)

    #for i in range(len(feature)-24*5):
        #_task(i)
        #time.sleep(1)
