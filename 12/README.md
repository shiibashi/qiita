## set up
apt install graphviz 

## Python
3.7.3

## 実行順
- cd script
- python make_dataset.py
- python train_script.py
- python vectorize_script.py --model data/ccae/best_model.h5

上記実行後、
- decode_script.ipynb: decoderの結果確認
- similarity_search_script.ipynb: 類似チャート探索

## 学習済みモデル

model/open_model.h5