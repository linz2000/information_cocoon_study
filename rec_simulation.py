from model_interface import Content_Rec_Interface, FM_Rec_Interface, NGCF_Rec_Interface
from config import BaseConfig, model_name, dataset_name
from dataset import BaseDataset, FMDataset, NGCF_Dataset
from model.DPP.dpp import dpp
from model.MMR.mmr import mmr, mmr_new

import pandas as pd
import numpy as np
import math
import torch
import faiss
import os
import pickle as pkl
import random
from collections import Counter

def getTopKSim(ue, ie, k):
    index = faiss.IndexFlatIP(ue.shape[1])
    index.add(ie)
    D, I = index.search(ue, k) # I: item index array, D: distance array
    # print(I.shape)
    return D, I

def getRandomBeta(u, s=0.05):
    try:
        a = ((1 - u) / (s * s) - 1 / u) * u * u
        b = a * (1 / u - 1)
        return np.random.beta(a, b)
    except:
        return u

class NewsRec():
    def __init__(self, news_path, behaviors_path, test_behaviors_path, test_fm_behaviors_path,
                 train_ngcf_behaviors_dir, train_ngcf_behaviors_path, val_ngcf_behaviors_path, test_ngcf_behaviors_path,
                 user2int_path, news2int_path, news2emb_path, checkpoint_dir, candidate_news_path, sel_user_path):
        self.news_path = news_path
        self.behaviors_path = behaviors_path
        self.test_behaviors_path = test_behaviors_path
        self.test_fm_behaviors_path = test_fm_behaviors_path
        self.train_ngcf_behaviors_dir = train_ngcf_behaviors_dir
        self.train_ngcf_behaviors_path = train_ngcf_behaviors_path
        self.val_ngcf_behaviors_path = val_ngcf_behaviors_path
        self.test_ngcf_behaviors_path = test_ngcf_behaviors_path
        self.user2int_path = user2int_path
        self.news2int_path = news2int_path
        self.news2emb_path = news2emb_path
        self.candidate_news_path = candidate_news_path
        self.sel_user_path = sel_user_path

        self.behaviors, self.test_behaviors, self.user2news_set, self.user_list, self.user2his_num = self.__load_behaviors()
        self.origin_behaviors = self.behaviors.copy()
        self.news, self.news_list, self.news2emb_arr, self.news2cat = self.__load_news()
        self.user2int, self.news2int, self.user2feat_id_list, self.news2feat_id_list = self.__gen_fm_data()


        if model_name == 'NRMS' or model_name == 'LSTUR' or model_name == 'TANR' or model_name == 'NAML' or model_name == 'DKN':
            self.model = Content_Rec_Interface(news_path, behaviors_path, user2int_path, checkpoint_dir)
        elif model_name == 'DFM' or model_name == 'NFM' or model_name == 'NCF' or \
                model_name == 'DFM_ID' or model_name == 'NFM_ID':
            self.model = FM_Rec_Interface(checkpoint_dir)
        elif model_name == 'NGCF':
            self.ngcf_data_generator = NGCF_Dataset(train_ngcf_behaviors_dir, train_ngcf_behaviors_path,
                                               val_ngcf_behaviors_path, test_ngcf_behaviors_path)
            self.test_behaviors = self.ngcf_data_generator
            self.model = NGCF_Rec_Interface(self.ngcf_data_generator, checkpoint_dir)
        else:
            print(f"{model_name} not included!")
            exit()
        test_res = self.model.evaluate(self.test_behaviors)
        self.model_acc = test_res[1]
        print("init acc:", self.model_acc)

    def __load_behaviors(self):
        behaviors = pd.read_table(
            self.behaviors_path,
            header=None,
            usecols=[1, 3],
            names=['user', 'clicked_news'],
            index_col=0)

        if model_name == 'NRMS' or model_name == 'LSTUR' or model_name == 'TANR' or model_name == 'NAML' or model_name == 'DKN':
            test_behaviors = BaseDataset(self.test_behaviors_path, self.news_path)
        elif model_name == 'DFM' or model_name == 'NFM':
            test_behaviors = FMDataset.init_from_path(self.test_fm_behaviors_path)
        elif model_name == 'NCF' or model_name == 'DFM_ID' or model_name == 'NFM_ID':
            test_behaviors = FMDataset.init_from_path(self.test_fm_behaviors_path, id_only=True)
        elif model_name == 'NGCF':
            test_behaviors = None
        else:
            print(f"{model_name} not included!")
            exit()

        #user2news_set, news_set: click history
        user2news_set = dict()
        user2his_num = dict()
        for row in behaviors.itertuples():
            user = row.Index
            clicked_news_list = row.clicked_news.strip().split()
            user2news_set[user] = set(clicked_news_list)
            user2his_num[user] = len(clicked_news_list)

        if self.sel_user_path == None:
            user_list = behaviors.index.tolist()
        else:
            user_list = []
            with open(self.sel_user_path, "r") as uf:
                for line in uf:
                    user_list.append(line.strip())

        return behaviors, test_behaviors, user2news_set, user_list, user2his_num

    def __load_news(self):
        news = pd.read_table(self.news_path, header=0, index_col=0)
        news2cat = dict()

        for row in news.itertuples():
            news_name = row.Index
            news_cat = row.category
            news2cat[news_name] = news_cat

        # names=['id', 'category', 'subcategory', 'title',
        #       'abstract', 'title_entities', 'abstract_entities']

        news_list = []
        with open(self.candidate_news_path, 'r') as f:
            for line in f:
                news_list.append(line.strip())

        with open(self.news2emb_path, 'rb') as embf:
            news2emb_arr = pkl.load(embf)

        return news, news_list, news2emb_arr, news2cat

    def __get_user_center(self, news2emb, bias=None):
        user2center = {}

        for row in self.behaviors.itertuples():
            user = row.Index
            clicked_news_list = row.clicked_news.strip().split()

            if bias != None:
                clicked_news_list = clicked_news_list[:bias]

            emb_list = []
            for news in clicked_news_list:
                emb = news2emb[news]
                if emb.is_cuda:
                    emb_list.append(emb.cpu().numpy())
                else:
                    emb_list.append(emb.numpy())

            center = np.mean(emb_list, axis=0)
            user2center[user] = center

        return user2center

    def __gen_fm_data(self):

        user2feat_id_list = {}
        news2feat_id_list = {}

        user2int = dict(pd.read_table(self.user2int_path).values.tolist())
        for user, id in user2int.items():
            user2feat_id_list[user] = [id]

        news2int = dict(pd.read_table(self.news2int_path).values.tolist())
        news_df = pd.read_table(self.news_path,
                                header=0,
                                usecols=[0, 1, 2])
        for row in news_df.itertuples():
            news = row.id
            cat = row.category
            subcat = row.subcategory

            if model_name == 'NCF' or model_name == 'DFM_ID' or model_name == 'NFM_ID':
                news2feat_id_list[news] = [news2int[news]]
            else:
                news2feat_id_list[news] = [news2int[news], cat, subcat]

        return user2int, news2int, user2feat_id_list, news2feat_id_list

    def __get_rec_list(self, click_prob, I, users, rec_num):
        if click_prob.is_cuda:
            click_prob = click_prob.cpu().numpy()
        else:
            click_prob = click_prob.numpy()

        idx = np.argsort(click_prob, axis=-1 )
        idx_arrays = idx[:, ::-1]

        # _, idx = torch.sort(click_prob, dim=-1, descending=True)
        # if idx.is_cuda:
        #     idx_arrays = idx.cpu().numpy()
        # else:
        #     idx_arrays = idx.numpy()
        news_index = [[row_I[i] for i in row_idx]
                      for row_idx, row_I in zip(idx_arrays, I)]

        rec_news_list = []
        rec_prob_list = []
        for i, user in enumerate(users):
            news_list_tmp = []
            prob_list_tmp = []

            for j, news_idx in enumerate(news_index[i]):
                news = self.news_list[news_idx]
                prob = click_prob[i][idx_arrays[i][j]]

                if news not in self.user2news_set[user]:
                    news_list_tmp.append(news)
                    prob_list_tmp.append(prob)
                    if news_list_tmp.__len__() >= rec_num:
                        break
            rec_news_list.append(news_list_tmp)
            rec_prob_list.append(prob_list_tmp)

        return rec_news_list, rec_prob_list

    def __get_rec_list_dpp(self, click_prob, I, users, rec_num, news_arrays):

        rec_news_list = []
        rec_prob_list = []

        click_prob = click_prob.tolist()
        for i, user in enumerate(users):
            click_prob_tmp = click_prob[i]
            I_tmp = I[i]
            click_prob_list = []
            idx_list = []
            for prob, idx in zip(click_prob_tmp, I_tmp):
                if self.news_list[idx] not in self.user2news_set[user]:
                    click_prob_list.append(prob)
                    idx_list.append(idx)

            np_scores = np.array(click_prob_list)
            np_feats = np.array([news_arrays[idx] for idx in idx_list])
            np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)
            similarities = np.dot(np_feats, np_feats.T)
            candidate_num = len(idx_list)
            kernel_matrix = np_scores.reshape((candidate_num, 1)) * similarities * np_scores.reshape((1, candidate_num))
            rec_idx_list = dpp(kernel_matrix, rec_num)

            news_list_tmp = [self.news_list[idx_list[idx]] for idx in rec_idx_list]
            prob_list_tmp = [click_prob_list[idx] for idx in rec_idx_list]

            rec_news_list.append(news_list_tmp)
            rec_prob_list.append(prob_list_tmp)

        return rec_news_list, rec_prob_list

    def __get_rec_list_mmr(self, click_prob, I, users, rec_num, news_arrays):

        rec_news_list = []
        rec_prob_list = []

        click_prob = click_prob.tolist()
        for i, user in enumerate(users):
            click_prob_tmp = click_prob[i]
            I_tmp = I[i]
            click_prob_list = []
            idx_list = []
            for prob, idx in zip(click_prob_tmp, I_tmp):
                if self.news_list[idx] not in self.user2news_set[user]:
                    click_prob_list.append(prob)
                    idx_list.append(idx)

            np_feats = np.array([news_arrays[idx] for idx in idx_list])
            np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)
            similarities = np.dot(np_feats, np_feats.T)
            rec_idx_list = mmr(click_prob_list, similarities, rec_num, lambda_constant=0.5)

            news_list_tmp = [self.news_list[idx_list[idx]] for idx in rec_idx_list]
            prob_list_tmp = [click_prob_list[idx] for idx in rec_idx_list]

            rec_news_list.append(news_list_tmp)
            rec_prob_list.append(prob_list_tmp)

        return rec_news_list, rec_prob_list

    def __get_rec_list_mmr_new(self, click_prob, I, users, rec_num, news_arrays, user2center, lambda_para=0.5):

        rec_news_list = []
        rec_prob_list = []

        click_prob = click_prob.tolist()
        for i, user in enumerate(users):
            click_prob_tmp = click_prob[i]
            I_tmp = I[i]
            click_prob_list = []
            idx_list = []
            for prob, idx in zip(click_prob_tmp, I_tmp):
                if self.news_list[idx] not in self.user2news_set[user]:
                    click_prob_list.append(prob)
                    idx_list.append(idx)

            np_feats = np.array([news_arrays[idx] for idx in idx_list])
            np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)

            user_feat = user2center[user]
            user_feat /= np.linalg.norm(user_feat)
            user_item_sim = [np.dot(user_feat, item_feat) for item_feat in np_feats]

            rec_idx_list = mmr_new(click_prob_list, user_item_sim, rec_num, lambda_constant=lambda_para)

            news_list_tmp = [self.news_list[idx_list[idx]] for idx in rec_idx_list]
            prob_list_tmp = [click_prob_list[idx] for idx in rec_idx_list]

            rec_news_list.append(news_list_tmp)
            rec_prob_list.append(prob_list_tmp)

        return rec_news_list, rec_prob_list

    def __simulate_click(self, rec_news_list, rec_prob_list):

        click_news_list = []

        for i in range(rec_news_list.__len__()):
            tmp = []
            for j, prob in enumerate(rec_prob_list[i]):
                click_prob = getRandomBeta(prob) # beta random
                sim_click_prob = self.model_acc * click_prob # simulate user true click prob
                random_p = np.random.random()
                if random_p < sim_click_prob:
                    tmp.append(rec_news_list[i][j])
            click_news_list.append(tmp)

        return click_news_list

    def update_click_history(self, user2added_news):

        for user in self.behaviors.index:
            if user in user2added_news:
                added_news = user2added_news[user]
                if added_news.__len__() > 0:
                    added_news_str = ' '.join(added_news)
                    self.behaviors.loc[user, 'clicked_news'] = added_news_str + ' ' +\
                                                self.behaviors.loc[user, 'clicked_news']

        for user in self.user2news_set:
            if user in user2added_news:
                self.user2news_set[user] = self.user2news_set[user].union(
                    set(user2added_news[user]) )

        for user in self.user2his_num:
            if user in user2added_news:
                self.user2his_num[user] = self.user2his_num[user] + len(user2added_news[user])

    def __get_neg_news(self, user, neg_num):
        neg_news_list = []

        news_num = len(self.news_list)
        while len(neg_news_list) < neg_num:
            news = self.news_list[random.randint(0, news_num - 1)]
            if news not in neg_news_list and news not in self.user2news_set[user]:
                neg_news_list.append(news)

        return neg_news_list

    def __get_ngcf_train_data(self, user2added_news):

        added_users = []
        added_pos_items = []
        added_neg_items = []

        for user, added_news in user2added_news.items():
            neg_num = len(added_news)
            neg_news_list = self.__get_neg_news(user, neg_num)

            for i in range(neg_num):
                added_users.append(self.user2int[user]-1)
                added_pos_items.append(self.news2int[added_news[i]]-1)
                added_neg_items.append(self.news2int[neg_news_list[i]]-1)

        return added_users, added_pos_items, added_neg_items

    def __get_fm_train_data(self, user2added_news):

        target = []
        feat_id_list = []

        for user, added_news in user2added_news.items():
            neg_num = len(added_news)
            neg_news_list = self.__get_neg_news(user, neg_num)

            for i in range(neg_num):
                feat_id_list.append(self.user2feat_id_list[user] + self.news2feat_id_list[added_news[i]] )
                feat_id_list.append(self.user2feat_id_list[user] + self.news2feat_id_list[neg_news_list[i]])
                target.append(1)
                target.append(0)

        return feat_id_list, target

    def __get_content_rec_train_data(self, added_news_history, user2added_news):
        data = []

        for user, added_news in user2added_news.items():
            neg_num = len(added_news)
            neg_news_list = self.__get_neg_news(user, neg_num)

            user_id = self.user2int[user]
            origin_clicked_news = self.origin_behaviors.loc[user, 'clicked_news']
            added_news_list = []
            for record in added_news_history[::-1]:
                if user in record:
                    added_news_list.extend(record[user])
            clicked_news = origin_clicked_news
            if added_news_list.__len__() > 0:
                clicked_news =  ' '.join(added_news_list) + ' ' + clicked_news

            for i in range(neg_num):
                candidate_news = [added_news[i], neg_news_list[i]]
                clicked = ['1', '0']
                data.append([user_id, clicked_news, ' '.join(candidate_news), ' '.join(clicked)])

        behaviors_parsed = pd.DataFrame(data, columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

        return behaviors_parsed

    def rec(self, recall_num=500, rec_num=50, rec_round=6, update_interval=1, post_process=None):
        batch_size = BaseConfig.batch_size
        user_num = len(self.user_list)
        news_num = len(self.news_list)
        print("user num:", user_num)
        print("news num:", news_num)

        if not os.path.exists(f"res/{dataset_name}/{model_name}"):
            os.makedirs(f"res/{dataset_name}/{model_name}")

        added_news_records = []
        complete_news_records = []

        for round_i in range(rec_round):
            print("rec round:", round_i + 1)

            # get all news embbeddings
            if model_name == 'NRMS' or model_name == 'LSTUR' or model_name == 'TANR' or model_name == 'NAML' or model_name == 'DKN':
                news2emb = self.model.getNews2Emb()
            elif model_name == 'DFM' or model_name == 'NFM' or model_name == 'NCF' or \
                    model_name == 'DFM_ID' or model_name == 'NFM_ID':
                news2emb = self.model.getNews2Emb(self.news2feat_id_list)
            elif model_name == 'NGCF':
                news2emb = self.model.getNews2Emb(self.news2int)
            else:
                print(f"{model_name} not included!")
                exit()

            news_tensors = torch.stack([news2emb[news] for news in self.news_list], dim=0)
            if news_tensors.is_cuda:
                news_arrays = news_tensors.cpu().numpy()
            else:
                news_arrays = news_tensors.numpy()
            news_arrays = news_arrays.astype('float32')


            if model_name == 'NRMS' or model_name == 'LSTUR' or model_name == 'TANR' or model_name == 'NAML' or model_name == 'DKN':
                user2emb = self.model.getUser2Emb(news2emb)
            elif model_name == 'DFM' or model_name == 'NFM' or model_name == 'NCF' or \
                    model_name == 'DFM_ID' or model_name == 'NFM_ID':
                user2emb = self.model.getUser2Emb(self.user2feat_id_list)
            elif model_name == 'NGCF':
                user2emb = self.model.getUser2Emb(self.user2int)
            else:
                print(f"{model_name} not included!")
                exit()

            user2center = self.__get_user_center(news2emb)
            user2added_news = dict()

            for i in range(math.ceil(user_num/batch_size)):
                users = self.user_list[i*batch_size: (i+1)*batch_size]

                user_tensors = torch.stack([user2emb[u] for u in users], dim=0)
                if model_name != 'DKN':
                    if user_tensors.is_cuda:
                        user_arrays = user_tensors.cpu().numpy()
                    else:
                        user_arrays = user_tensors.numpy()
                else:
                    mean_user_tensors = torch.stack([torch.mean(user2emb[u], dim=0) for u in users], dim=0)
                    if mean_user_tensors.is_cuda:
                        user_arrays = mean_user_tensors.cpu().numpy()
                    else:
                        user_arrays = mean_user_tensors.numpy()
                user_arrays = user_arrays.astype('float32')

                # recall
                # I: index, batch_size x recall_num
                D, I = getTopKSim(user_arrays, news_arrays, recall_num)

                # get click probability
                if model_name == 'NRMS' or model_name == 'LSTUR' or model_name == 'TANR' or model_name == 'NGCF' or model_name == 'NAML' or model_name == 'DKN':

                    # batch_size x recall_num x news_emb_dim
                    recall_news_tensors = torch.stack([ torch.stack([news_tensors[i] for i in row], dim=0)
                                                        for row in I], dim=0)
                    # click_prob: batch_size x recall_num
                    click_prob = self.model.getChickProb(recall_news_tensors, user_tensors)

                elif model_name == 'DFM' or model_name == 'NFM' or model_name == 'NCF' or \
                        model_name == 'DFM_ID' or model_name == 'NFM_ID':
                    user_feat = [] # batch_size of list
                    for user in users:
                        user_feat.append(self.user2feat_id_list[user])

                    recall_news_feat = [] # batch_size x recall_num of list
                    for row in I:
                        tmp = []
                        for news_id in row:
                            tmp.append( self.news2feat_id_list[self.news_list[news_id]])
                        recall_news_feat.append(tmp)

                    # click_prob: batch_size x recall_num
                    click_prob = self.model.getChickProb(user_feat, recall_news_feat)
                else:
                    print(f"{model_name} not included!")
                    exit()

                # batch_size x rec_num, list of list
                if post_process == "dpp":
                    rec_news_list, rec_prob_list = self.__get_rec_list_dpp(click_prob, I, users, rec_num, news_arrays)
                elif post_process == "mmr":
                    rec_news_list, rec_prob_list = self.__get_rec_list_mmr(click_prob, I, users, rec_num, news_arrays)
                elif post_process.startswith("new"): # ICMF
                    lambda_para = float(post_process.split('_')[-1])
                    rec_news_list, rec_prob_list = self.__get_rec_list_mmr_new(click_prob, I, users, rec_num, news_arrays, user2center, lambda_para=lambda_para)
                else:
                    rec_news_list, rec_prob_list = self.__get_rec_list(click_prob, I, users, rec_num)

                # batch_size x { news_list }
                click_news_list = self.__simulate_click(rec_news_list, rec_prob_list)

                # record newly added news
                user2added_news.update(dict(zip(users, click_news_list)))
            print("done")
            # update dataset
            self.update_click_history(user2added_news)
            added_news_records.append(user2added_news)
            complete_news_records.append(user2added_news)

            if (round_i+1) % update_interval == 0:  # update model
                if model_name == 'NRMS' or model_name == 'LSTUR' or model_name == 'TANR' or model_name == 'NAML' or model_name == 'DKN':
                    for i, record in enumerate(added_news_records):
                        self.model.updateUserDataset(record)
                        new_behaviors = self.__get_content_rec_train_data(complete_news_records[:(-1)*update_interval + i], record)
                        self.model.retrain(new_behaviors)
                elif model_name == 'DFM' or model_name == 'NFM' or model_name == 'NCF' or \
                        model_name == 'DFM_ID' or model_name == 'NFM_ID':
                    for record in added_news_records:
                        feat_id_list, target = self.__get_fm_train_data(record)
                        self.model.retrain(feat_id_list, target)
                elif model_name == 'NGCF':
                    for record in added_news_records:
                        added_data = self.__get_ngcf_train_data(record)
                        self.model.retrain(added_data)

                auc, acc, macro_f1, micro_f1 = self.model.evaluate(self.test_behaviors)
                print(f"rec round {round_i + 1} res:", auc, acc, macro_f1, micro_f1)
                self.model_acc = acc

                added_news_records = []

            # save to file
            if post_process != None:
                save_path = f"res/{dataset_name}/{model_name}/behaviors_recall_{recall_num}_rec_{rec_num}_interval_{update_interval}_{round_i+1}_post_process_{post_process}.tsv"
            else:
                save_path = f"res/{dataset_name}/{model_name}/behaviors_recall_{recall_num}_rec_{rec_num}_interval_{update_interval}_{round_i + 1}.tsv"
            self.behaviors.to_csv(save_path, sep="\t", header=False, index=True)


def main():
    print("model name:", model_name)

    post_process = None # 'dpp', 'mmr', 'new_0.5'

    behaviors_path = f"data/{dataset_name}/test/behaviors.tsv"
    test_behaviors_path = f"data/{dataset_name}/test/behaviors_parsed.tsv"
    test_fm_behaviors_path = f"data/{dataset_name}/test/fm_behaviors.tsv"
    test_ngcf_behaviors_path = f"data/{dataset_name}/test/ngcf_behaviors.txt"

    train_ngcf_behaviors_dir = f"data/{dataset_name}/train"
    train_ngcf_behaviors_path = f"data/{dataset_name}/train/ngcf_behaviors.txt"
    val_ngcf_behaviors_path = f"data/{dataset_name}/val/ngcf_behaviors.txt"

    news_path = f"data/{dataset_name}/train/news_parsed.tsv"
    news2int_path = f"data/{dataset_name}/train/news2int.tsv"
    user2int_path = f"data/{dataset_name}/train/user2int.tsv"

    checkpoint_dir = f"checkpoint/{dataset_name}/{model_name}"
    news2emb_path = f"data/{dataset_name}/train/news2emb.pkl"

    candidate_news_path = f"data/{dataset_name}/test/candidate_news.txt"
    sel_user_path = None


    news_recommender = NewsRec(news_path, behaviors_path, test_behaviors_path, test_fm_behaviors_path,
                               train_ngcf_behaviors_dir, train_ngcf_behaviors_path, val_ngcf_behaviors_path, test_ngcf_behaviors_path,
                               user2int_path, news2int_path, news2emb_path, checkpoint_dir, candidate_news_path, sel_user_path)
    news_recommender.rec(post_process=post_process)



if __name__ == '__main__':
    main()