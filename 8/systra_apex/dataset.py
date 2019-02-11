import pandas
import numpy

N = 1000


def make_dataset():
    df = pandas.DataFrame(
        {"x": numpy.random.rand(N), "y": numpy.random.rand(N)}
    )
    profit = []
    for x, y in zip(df["x"], df["y"]):
        r = numpy.random.rand()
        if x >= y * 2:
            profit.append(r)
        elif x >= y and r >= 0.5:
            profit.append(r)
        else:
            profit.append(-r)
    df["profit"] = pandas.Series(profit)
    return df
