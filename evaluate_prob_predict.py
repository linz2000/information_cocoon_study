from config import model_name, dataset_name
from dataset import BaseDataset

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from multiprocessing import Pool

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from config import device

try:
    if model_name in model_name in ['NRMS', 'LSTUR', 'TANR', 'NAML', 'DKN']:
        Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
        config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}


class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item


class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR':
            print(f'User miss rate: {user_missed/user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()
        # list of str

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4

def calculate_f1_score(pair):
    try:
        macro_f1 = f1_score(*pair, average='macro')
        micro_f1 = f1_score(*pair, average='micro')
        return [macro_f1, micro_f1]
    except ValueError:
        return [np.nan] * 2

def test_content_rec_model(model, dataset: BaseDataset):
    model.eval()

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=False,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=False))

    auc_list, acc_list, macro_f1_list, micro_f1_list = [], [], [], []

    with torch.no_grad():
        for i in tqdm(range(1, len(dataset) // config.batch_size + 1), desc="Testing"):
            minibatch = next(dataloader)

            if model_name == 'LSTUR':
                y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                               minibatch["candidate_news"],
                               minibatch["clicked_news"])
            elif model_name == 'HiFiArk':
                y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                                 minibatch["clicked_news"])
            elif model_name == 'TANR':
                y_pred, topic_classification_loss = model(
                    minibatch["candidate_news"], minibatch["clicked_news"])
            else:
                y_pred = model(minibatch["candidate_news"],
                               minibatch["clicked_news"])

            sigmod = nn.Sigmoid()
            y_pred = sigmod(y_pred)  # batchszie x sample num per row
            y_pred = y_pred.reshape(-1)
            if y_pred.is_cuda:
                y_pred = y_pred.cpu().numpy()
            else:
                y_pred = y_pred.numpy()
            y_pred_l = np.array([1 if s > 0.5 else 0 for s in y_pred])
            y = torch.stack(minibatch['clicked'], dim=0).t().float()
            y = y.reshape(-1).numpy()

            # print(y.shape)
            # print(y_pred.shape)

            macro_f1 = f1_score(y, y_pred_l, average='macro')
            micro_f1 = f1_score(y, y_pred_l, average='micro')
            auc = roc_auc_score(y, y_pred)
            acc = accuracy_score(y, y_pred_l)

            auc_list.append(auc)
            acc_list.append(acc)
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)

    return np.mean(auc_list), np.mean(acc_list), np.mean(macro_f1_list), np.mean(micro_f1_list)

@torch.no_grad()
def evaluate(model, directory, num_workers, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        num_workers: processes number for calculating metrics
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    # news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
    news_dataset = NewsDataset(path.join(f'data/{dataset_name}/train/news_parsed.tsv') )
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=False)

    news2vector = {}
    for minibatch in tqdm(news_dataloader,
                          desc="Calculating vectors for news"):
        news_ids = minibatch["id"]

        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'),
                               f'data/{dataset_name}/train/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=False)

    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                          desc="Calculating vectors for users"):

        user_strings = minibatch["clicked_news_string"]   # num_clicked_news_a_user x batch_size, two layers of dict
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                              dim=0).transpose(0, 1)

            # batch_size x num_clicked_news_a_user x user embeddding dim
            # print(clicked_news_vector.shape)

            if model_name == 'LSTUR':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    clicked_news_vector)
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))  # since different length of  impressions, batch_size only 1
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0

    tasks = []

    for minibatch in tqdm(behaviors_dataloader,
                          desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack([   #impression_num x new embedding dim
            news2vector[news[0].split('-')[0]]
            for news in minibatch['impressions']
        ],
                                            dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]] #[user embedding dim, ]
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        # y_pred = click_probability.tolist()
        sigmod = nn.Sigmoid()
        y_pred = sigmod(click_probability).tolist()

        y_true = [
            int(news[0].split('-')[1]) for news in minibatch['impressions']
        ]

        tasks.append((y_true, y_pred))

    pred_label_list = []
    true_label_list = []

    for y_true_l, y_score_l in tasks:

        y_pred_l = [1 if s > 0.5 else 0 for s in y_score_l]
        pred_label_list.extend(y_pred_l)

        true_label_list.extend(y_true_l)


    with Pool(processes=num_workers+1) as pool:        # windows: num_worker only 0
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T

    macro_f1 = f1_score(true_label_list, pred_label_list, average='macro')
    micro_f1 = f1_score(true_label_list, pred_label_list, average='micro')

    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(
        ndcg10s), macro_f1, micro_f1


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train_prob_predict import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join(f'./checkpoint/{dataset_name}', model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    auc, mrr, ndcg5, ndcg10, macro_f1, micro_f1 = evaluate(model, f'./data/{dataset_name}/test',
                                       config.num_workers)
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}\nmacro f1: {macro_f1:.4f}\nmicro f1: {micro_f1:.4f}'
    )
