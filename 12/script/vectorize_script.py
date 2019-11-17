import train.vectorize_with_encoder
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    train.vectorize_with_encoder.run(
        args.model,
        #"data/cae/best_model.h5",
        "data/img/clustering", "data/clustering_vec.csv")
