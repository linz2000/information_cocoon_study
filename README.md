# information_cocoon_study
Source code of the paper "An Exploratory Study on Information Cocoon in Recommender Systems".

## 1. running details

- config.py -- basic configuration: dataset, model, cuda, etc.

- rec_simulation.py --  interaction simulation between user and news recommender system.

## 2. train models

Before running 'rec_simulation.py', the recommendation model needs to be trained in advance.
- train_prob_predict.py -- train news recommendation model: NRMS, NAML, DKN.
- train_fm_and_ncf.py -- train DeepFM and NCF models.
- train_ngcf.py -- train NGCF model.

And the model implementation code is in 'model/' directory.

## 3. datasets

You can get the preprocessed data through Baidu Cloud Disk:

Link: https://pan.baidu.com/s/1tsW6CqFbG8OMYT1aTHhEQQ

Extraction code: kj7a

## 4. references

recommendation models refer to the following:

- NGCF: [huangtinglin/NGCF-PyTorch: PyTorch Implementation for Neural Graph Collaborative Filtering (github.com)](https://github.com/huangtinglin/NGCF-PyTorch)

- NCF and DeepFM: [rixwew/pytorch-fm: Factorization Machine models in PyTorch (github.com)](https://github.com/rixwew/pytorch-fm)

- NRMS, NAML and DKN: [yusanshi/news-recommendation: Implementations of some methods in news recommendation. (github.com)](https://github.com/yusanshi/news-recommendation)

- DPP: [laming-chen/fast-map-dpp: Fast Greedy MAP Inference for DPP. (github.com)](https://github.com/laming-chen/fast-map-dpp)

- DGRec: [YangLiangwei/DGRec (github.com)](https://github.com/YangLiangwei/DGRec)
