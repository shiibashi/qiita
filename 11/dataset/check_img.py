import pandas
import cv2

def load_dataset():
    df = pandas.read_csv("feature.csv")
    train_df = df[10000:20000].reset_index(drop=True)
    test_df = df[20000:].reset_index(drop=True)
    return train_df, test_df

if __name__ == "__main__":
    img_list = []
    train_df, test_df = load_dataset()
    for t in train_df["Timestamp"]:
        img = cv2.imread("img/{}.png".format(t))
        if img is None:
            img_list.append(t)
        #assert img is not None
    for t in test_df["Timestamp"]:
        img = cv2.imread("img/{}.png".format(t))
        if img is None:
            img_list.append(t)
    print(img_list)
