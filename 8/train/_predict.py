from keras.preprocessing.image import ImageDataGenerator
import network
import _parameter as p

from PIL import Image
import os
import numpy
import pandas

DIR_PATH = os.path.dirname(__file__)

def run():
    model = _load_model()
    prob_1_list = []
    label_list = []
    
    dirname = "{}/img/test/label_1".format(DIR_PATH)
    for filename in os.listdir(dirname):
        img = Image.open("{}/{}".format(dirname, filename))
        img_rgb = img.convert("RGB")
        arr = numpy.array(img_rgb).reshape((1, p.IMG_ROWS, p.IMG_COLS, 3))
        pred = model.predict(arr)
        prob_1 = pred[0][1]
        prob_1_list.append(prob_1)
        label_list.append("label_1")

    dirname = "{}/img/test/label_0".format(DIR_PATH)
    for filename in os.listdir(dirname):
        img = Image.open("{}/{}".format(dirname, filename))
        img_rgb = img.convert("RGB")
        arr = numpy.array(img_rgb).reshape((1, p.IMG_ROWS, p.IMG_COLS, 3))
        pred = model.predict(arr)
        prob_1 = pred[0][1]
        prob_1_list.append(prob_1)
        label_list.append("label_0")
    df = pandas.DataFrame({"prob": prob_1_list, "label": label_list})
    df.to_csv("predict.csv", index=False)

    print(">=0.95")
    print(df.query("prob >= 0.95")["label"].value_counts())
    print(">=0.9")
    print(df.query("prob >= 0.9")["label"].value_counts())
    print(">=0.8")
    print(df.query("prob >= 0.8")["label"].value_counts())
    print(">=0.7")
    print(df.query("prob >= 0.7")["label"].value_counts())
    print(">=0.6")
    print(df.query("prob >= 0.6")["label"].value_counts())
    print(">=0.5")
    print(df.query("prob >= 0.5")["label"].value_counts())
    print("<=2")
    print(df.query("prob <= 0.2")["label"].value_counts())
    print("<=1")
    print(df.query("prob <= 0.1")["label"].value_counts())

def _load_model():
    model = network.ResnetBuilder.build_resnet_18((p.CHANNELS, p.IMG_ROWS, p.IMG_COLS), p.CLASSES)
    model.load_weights("{}/models/best_model.h5".format(DIR_PATH))
    return model
