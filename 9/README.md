# Qiita
https://qiita.com/shiibass/items/167cffabaccddc5d8f1a

# BERT as service

```
mkdir model
cd model
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
cd ..
bert-serving-start -model_dir model/cased_L-12_H-768_A-12/ \
    -num_worker=1\
    -max_seq_len=300
```

# Python
3.6.5
```
pip install -r requirement.txt
```