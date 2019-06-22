from bert_serving.client import BertClient
import pandas
import numpy

if __name__ == "__main__":
    df = pandas.read_csv("news.csv")
    with BertClient(ip="0.0.0.0") as client:
        vecs = client.encode(list(df["News"]))
    numpy.savetxt("news_vec.csv", vecs, delimiter=",")