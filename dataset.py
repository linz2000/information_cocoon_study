from config import model_name, MINDConfig, NGCFConfig

from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
import importlib
import torch
import pickle as pkl
import scipy.sparse as sp
from time import time
import random as rd


try:
    if model_name in model_name in ['NRMS', 'LSTUR', 'TANR', 'NAML', 'DKN']:
        config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, behaviors_mode='file', behaviors_pd: pd.DataFrame=None):
        super(BaseDataset, self).__init__()
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])

        if behaviors_mode == 'file':
            self.behaviors_parsed = pd.read_table(behaviors_path)
        else:
            self.behaviors_parsed = behaviors_pd

        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(    # to tensor
                    self.news2dict[key1][key2])
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])   # to tensor

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item

    # return item
    # a dict of: user(id) clicked(int 0/1 list)
    # candidate_news(list of news [dict for news type, tensor for item type])
    # clicked_news (list of news [dict for news type, tensor for item type])

class NewsDataset(Dataset):
    """
    Load news.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()

        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })

        self.news2cat, self.news2subcat = self.__gen_category_dict()

        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __gen_category_dict(self):
        new2category = dict()
        new2subcategory = dict()
        for row in self.news_parsed.itertuples():
            news = row.id
            category = row.category
            subcategory = row.subcategory

            new2category[news] = category
            new2subcategory[news] = subcategory

        return new2category , new2subcategory

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item

    def getNewsCat(self, news):
        if news not in self.news2cat:
            print(news)
            print("not in")
            return 0
        return self.news2cat[news]

    def getNewsSubcat(self, news):
        if news not in self.news2subcat:
            print(news)
            print("not in")
            return 0
        return self.news2subcat[news]

    def getNews2cat(self):
        return self.news2cat

    def getNews2subcat(self):
        return self.news2subcat

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path, behaviors_dataframe: pd.DataFrame=None):
        super(UserDataset, self).__init__()
        if behaviors_dataframe is not None:
            self.behaviors = behaviors_dataframe[['user', 'clicked_news']]
        else:
            self.behaviors = pd.read_table(behaviors_path,
                                           header=None,
                                           usecols=[1, 3],
                                           names=['user', 'clicked_news'])
            self.behaviors.clicked_news.fillna(' ', inplace=True)
            self.behaviors.drop_duplicates(inplace=True)

        self.user2int = dict(pd.read_table(user2int_path).values.tolist())

        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in self.user2int:
                # self.behaviors.at[row.Index, 'user'] = self.user2int[row.user]  # no editing 'user'
                pass
            else:
                user_missed += 1
                # self.behaviors.at[row.Index, 'user'] = 0
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
            row.clicked_news.split()[:config.num_clicked_news_a_user],
            "id":
            self.user2int[row.user] if row.user in self.user2int else 0 # add
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item

    def updateDataset(self, user2added_news):

        for idx in self.behaviors.index:
            user = self.behaviors.loc[idx, 'user']
            if user in user2added_news:
                added_news = user2added_news[user]
                if added_news.__len__() > 0:
                    added_news_str = ' '.join(added_news)
                    self.behaviors.loc[idx, "clicked_news"] = added_news_str + ' ' +\
                                                self.behaviors.loc[idx, "clicked_news"]

class FMDataset():
    def __init__(self, items, targets):
        self.items = items
        self.targets = targets

    @classmethod
    def init_from_path(cls, behaviors_path, id_only=False):
        data = pd.read_table(
            behaviors_path,
            header=0).to_numpy()
        if id_only:
            items = data[:, :2].astype(np.int)
        else:
            items = data[:, :4].astype(np.int)
        targets = data[:, 4].astype(np.float32)

        return cls(items, targets)

    @classmethod
    def init_from_list(cls, items, targets):
        items = np.array(items).astype(np.int)
        targets = np.array(targets).astype(np.float32)

        return cls(items, targets)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

class NGCF_Dataset(object):
    def __init__(self, dir, train_path, val_path, test_path, batch_size=NGCFConfig.batch_size):
        self.path = dir
        self.batch_size = batch_size

        train_file = train_path
        val_file = val_path
        test_file = test_path

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_val, self.n_test = 0, 0, 0

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_val += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.val_items, self.test_items = {}, {}, {} # uid -> neig node list
        self.train_items_set, self.val_items_set, self.test_items_set = {}, {}, {}

        with open(train_file) as f_train:
            with open(val_file) as f_val:
                with open(test_file) as f_test:
                    for l in f_train.readlines():
                        if len(l) == 0:
                            break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, train_items = items[0], items[1:]

                        for i in train_items:
                            self.R[uid, i] = 1.
                            # self.R[uid][i] = 1

                        self.train_items[uid] = train_items
                        self.train_items_set[uid] = set(train_items)

                    for l in f_val.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        try:
                            items = [int(i) for i in l.split(' ')]
                        except Exception:
                            continue

                        uid, val_items = items[0], items[1:]
                        self.val_items[uid] = val_items
                        self.val_items_set[uid] = set(val_items)

                    for l in f_test.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        try:
                            items = [int(i) for i in l.split(' ')]
                        except Exception:
                            continue

                        uid, test_items = items[0], items[1:]
                        self.test_items[uid] = test_items
                        self.test_items_set[uid] = set(test_items)

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        # adj_mat = adj_mat.tolil()
        # R = self.R.tolil()
        #
        # adj_mat[:self.n_users, self.n_users:] = R
        # adj_mat[self.n_users:, :self.n_users] = R.T
        # adj_mat = adj_mat.todok()

        row, col = self.R.nonzero()
        for i, j in zip(row, col):
            adj_mat[i, j+self.n_users] = self.R[i, j]
            adj_mat[j+self.n_users, i] = self.R[i, j]

        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample_pos_items_for_u(self, u, num):
        # sample num pos items for u-th user
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num:
                break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample_neg_items_for_u(self, u, num):
        # sample num neg items for u-th user
        neg_items = []
        while True:
            if len(neg_items) == num:
                break
            neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_id not in self.train_items_set[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []
        for u in users:
            pos_items += self.sample_pos_items_for_u(u, 1)
            neg_items += self.sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_val_data(self):
        users, pos_items, neg_items = [], [], []
        for user, item_set in self.val_items.items():
            for pos_item in item_set:
                users.append(user)
                pos_items.append(pos_item)
                neg_items.append(self.sample_neg_items_for_u(user, 1)[0])

        return users, pos_items, neg_items

    def get_test_data(self):
        users, pos_items, neg_items = [], [], []
        for user, item_set in self.test_items.items():
            for pos_item in item_set:
                users.append(user)
                pos_items.append(pos_item)
                neg_items.append(self.sample_neg_items_for_u(user, 1)[0])

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_val + self.n_test))
        print('n_train=%d, n_val=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_val, self.n_test, (self.n_train + self.n_val + self.n_test)/(self.n_users * self.n_items)))
