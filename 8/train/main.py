import os
import shutil
import argparse
import _train
import _predict

DIR_PATH = os.path.dirname(__file__)
if DIR_PATH != "":
    os.chdir(DIR_PATH)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int, required=False) 
    parser.add_argument("--workers", default=1, type=int, required=False)
    parser.add_argument("--onlyPredict", type=int, required=False) # 1のとき予測のみ
    parser.add_argument("--myLoss", default=0, type=int, required=False) # 1のときmyloss
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print("parse arg")
    args = parse()
    if args.onlyPredict == 1:
        _predict.run()
        import sys
        sys.exit()

    print("train")
    _train.run(args)

    print("predict")
    _predict.run()
