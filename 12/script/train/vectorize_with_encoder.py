import pandas
import numpy
import os
import cv2
#from sklearn.manifold import TSNE

from . import train_ccae_model
DIR_PATH = os.path.dirname(__file__)

def run(model_filepath, data_dirpath, save_path):
    os.chdir("{}/..".format(DIR_PATH)) # script/ に移動
    mf = MakeFeature(model_filepath)
    df = mf.to_vec(data_dirpath, hold_days=10)
    #df2 = mf.make_tsne_feature(df)
    #df2.to_csv(save_path, index=False)
    df.to_csv(save_path, index=False)

class MakeFeature(object):
    def __init__(self, model_filepath):
        self.candlestick_encoder = train_ccae_model.CandlestickEncoder(model_filepath)

    def to_vec(self, data_dirpath, hold_days):
        img_filepath_list = os.listdir(data_dirpath)
        def _generate():
            for img_filepath in img_filepath_list:
                filepath = "{}/{}".format(data_dirpath, img_filepath)
                code, from_ymd, to_ymd = img_filepath.split(".")[0].split("_")
                #code_data = pandas.read_csv("data/code_data/{}.csv".format(code))

                #table_feature_list = self.make_table_feature(code_data, code, from_ymd, to_ymd, hold_days)

                ccae_feature_list = self.make_ccae_feature(filepath)

                #margin_feature_list = self.make_margin_feature(code, to_ymd)

                #ranking_feature_list = self.make_ranking_feature(code, to_ymd)

                #market_feature_list = self.make_market_feature(to_ymd)

                filepath_list = [img_filepath]

                data_list = filepath_list + ccae_feature_list
                
                yield data_list
        data = list(_generate())
        ccae_columns = ["CCAE_{}".format(i) for i in range(self.candlestick_encoder.encoder_feature_length)]
        columns = ["filepath"] + ccae_columns
        df = pandas.DataFrame(data, columns=columns)
        return df

    def make_tsne_feature(self, df):
        df = df.copy()
        model = TSNE(n_components=3)
        ccae_columns = ["CCAE_{}".format(i) for i in range(self.candlestick_encoder.encoder_feature_length)]
        X = model.fit_transform(df[ccae_columns])
        tsne_df = pandas.DataFrame(X, columns=["tsne_1", "tsne_2", "tsne_3"])
        res = pandas.concat([df, tsne_df], axis=1)
        return res


    def make_ccae_feature(self, filepath):
        img_arr = cv2.imread(filepath, 0)
        shape = (1, self.candlestick_encoder.shape[0],
                self.candlestick_encoder.shape[1], self.candlestick_encoder.shape[2])
        arr = img_arr.reshape(shape)
        pred = self.candlestick_encoder.predict_encoder(arr/255.0)
        return list(pred[0])