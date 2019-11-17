import train.make_clustering_dataset


if __name__ == "__main__":
    train.make_clustering_dataset.run(100, with_volume=False)
    train.make_clustering_dataset.select_train_test(test_num=20)