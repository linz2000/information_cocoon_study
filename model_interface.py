from config import model_name, dataset_name, device, FMConfig, MINDConfig, NGCFConfig
from dataset import NewsDataset, UserDataset, BaseDataset, FMDataset, NGCF_Dataset
from train_prob_predict import latest_checkpoint
from train_fm_and_ncf import train as train_fm_model
from train_fm_and_ncf import test_fm_model
from train_ngcf import retrain as train_ngcf_model
from train_ngcf import test_ngcf_model
from train_prob_predict import train_content_rec_model
from evaluate_prob_predict import test_content_rec_model
from model.NGCF.parser import parse_args

import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import math

class FM_Rec_Interface():
    def __init__(self, checkpoint_dir):

        if model_name == 'NCF' or model_name == 'DFM_ID' or model_name == 'NFM_ID':
            self.field_dims = np.array(MINDConfig.id_field_dims)
            self.item_feat_idx = MINDConfig.item_idx
            self.user_feat_idx = MINDConfig.user_idx
        else:
            self.field_dims = np.array(MINDConfig.field_dims)
            self.item_feat_idx = MINDConfig.item_feat_idx
            self.user_feat_idx = MINDConfig.user_feat_idx

        try:
            if model_name[-2:] == 'ID':
                Model = getattr(importlib.import_module(f"model.FM.{model_name[:-3].lower()}"), model_name[:-3])
            else:
                Model = getattr(importlib.import_module(f"model.FM.{model_name.lower()}"), model_name)
        except AttributeError:
            print(f"{model_name} not included!")
            exit()

        if model_name == 'DFM' or model_name == 'DFM_ID':
            self.model = Model(self.field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2).to(device)
        elif model_name == 'NFM' or model_name == 'NFM_ID':
            self.model = Model(self.field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2)).to(device)
        elif model_name == 'NCF':
            self.model = Model(self.field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                     user_field_idx=np.array(self.user_feat_idx, dtype=np.long),
                                     item_field_idx=np.array(self.item_feat_idx, dtype=np.long)).to(device)
        self.__loadModel(checkpoint_dir)

    def __loadModel(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def retrain(self, items, targets):

        dataset = FMDataset.init_from_list(items, targets)
        data_loader = DataLoader(dataset, batch_size=FMConfig.batch_size, num_workers=8)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=FMConfig.learning_rate,
                                     weight_decay=FMConfig.weight_decay)

        train_fm_model(self.model, optimizer, data_loader, criterion, device)

        self.model.eval()

    def evaluate(self, fm_behaviors):

        data_loader = DataLoader(fm_behaviors, batch_size=FMConfig.batch_size, num_workers=8)

        auc, acc, macro_f1, micro_f1 = test_fm_model(self.model, data_loader, device)

        return auc, acc, macro_f1, micro_f1

    def getNews2Emb(self, news2feat_id_list):
        feat_id_arr = []
        news_list = []
        for news, feat_id_list in news2feat_id_list.items():
            feat_id_arr.append(feat_id_list)
            news_list.append(news)

        news2vector = {}
        batch_size = FMConfig.batch_size

        for i in range(math.ceil(news_list.__len__() / batch_size)):
            news_list_cut = news_list[i * batch_size: (i + 1) * batch_size]

            feat_id_arr_cut = feat_id_arr[i * batch_size: (i + 1) * batch_size]
            feat_id_arr_cut = np.array(feat_id_arr_cut).astype(np.int)
            feat_id_tensor_cut = torch.LongTensor(feat_id_arr_cut).to(device)

            with torch.no_grad():
                news_embs_cut = self.model.get_embs(feat_id_tensor_cut, self.item_feat_idx)

            for news, emb in zip(news_list_cut, news_embs_cut):
                news2vector[news] = emb

        return news2vector


    def getUser2Emb(self, user2feat_id_list):
        feat_id_arr = []
        user_list = []
        for user, feat_id_list in user2feat_id_list.items():
            feat_id_arr.append(feat_id_list)
            user_list.append(user)

        user2vector = {}
        batch_size = FMConfig.batch_size

        for i in range(math.ceil(user_list.__len__() / batch_size)):
            user_list_cut = user_list[i*batch_size: (i+1)*batch_size]

            feat_id_arr_cut = feat_id_arr[i*batch_size: (i+1)*batch_size]
            feat_id_arr_cut = np.array(feat_id_arr_cut).astype(np.int)
            feat_id_tensor_cut = torch.LongTensor(feat_id_arr_cut).to(device)

            with torch.no_grad():
                user_embs_cut = self.model.get_embs(feat_id_tensor_cut, self.user_feat_idx)

            for user, emb in zip(user_list_cut, user_embs_cut):
                user2vector[user] = emb

        return user2vector

    def getChickProb(self, user_feat, recall_news_feat):

        input = []
        for u_feat, news_feat_list in zip(user_feat, recall_news_feat):
            for news_feat in news_feat_list:
                input.append(u_feat + news_feat)

        input = np.array(input).astype(np.int)
        input = torch.LongTensor(input).to(device)

        with torch.no_grad():
            pred = self.model(input) # (batch_size x candidate_size, )

        user_num = len(user_feat)
        pred = pred.reshape(user_num, -1)

        return pred

class NGCF_Rec_Interface():
    def __init__(self, data_generator: NGCF_Dataset, checkpoint_dir=None):

        try:
            Model = getattr(importlib.import_module(f"model.NGCF.{model_name.lower()}"), model_name)
        except AttributeError:
            print(f"{model_name} not included!")
            exit()

        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
        self.args = self.__load_args()

        self.model = Model(data_generator.n_users,
                      data_generator.n_items,
                      norm_adj,
                      self.args).to(self.args.device)

        self.__loadModel(checkpoint_dir)

    def __load_args(self):
        args = parse_args()
        args.device = device
        args.batch_size = NGCFConfig.batch_size
        args.epoch = NGCFConfig.epoch
        args.weights_path = f'checkpoint/{dataset_name}/NGCF'
        args.node_dropout = eval(args.node_dropout)
        args.mess_dropout = eval(args.mess_dropout)

        return args

    def __loadModel(self, checkpoint_dir):
        if checkpoint_dir == None:
            checkpoint_path = os.path.join(self.args.weights_path, "ckpt.pth")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def retrain(self, added_data): # added_data:   added_users, added_pos_items, added_neg_items
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        train_ngcf_model(self.model, optimizer, added_data, criterion, self.args)

        self.model.eval()

    def evaluate(self, data_generator: NGCF_Dataset):
        auc, acc, macro_f1, micro_f1 = test_ngcf_model(self.model, data_generator)

        return auc, acc, macro_f1, micro_f1

    def getNews2Emb(self, news2int: dict): # id = id -1
        int2news = dict(zip(news2int.values(), news2int.keys()))

        news_id_list = [id-1 for id in news2int.values()]

        with torch.no_grad():
            _, news_embeddings, _ = self.model([], news_id_list, [], drop_flag=self.args.node_dropout_flag)

        news2emb = {}
        for id, emb in zip(news_id_list, news_embeddings):
            news2emb[ int2news[id+1] ] = emb

        return news2emb

    def getUser2Emb(self, user2int: dict): # id = id -1
        int2user = dict(zip(user2int.values(), user2int.keys()))

        user_id_list = [id-1 for id in user2int.values()]

        with torch.no_grad():
            user_embeddings, _, _ = self.model(user_id_list, [], [], drop_flag=self.args.node_dropout_flag)

        user2emb = {}
        for id, emb in zip(user_id_list, user_embeddings):
            user2emb[ int2user[id+1] ] = emb

        return user2emb

    def getChickProb(self, news_vector, user_vector):

        logits = torch.bmm(news_vector, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        sigmod = nn.Sigmoid()
        click_probability = sigmod(logits)
        return click_probability

class Content_Rec_Interface():
    def __init__(self, news_path, behaviors_path, user2int_path, checkpoint_dir):
        self.behaviors_path = behaviors_path
        self.news_path = news_path
        self.news_dataset = NewsDataset(news_path)
        self.user_dataset = UserDataset(behaviors_path, user2int_path)

        try:
            Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
            self.config = getattr(importlib.import_module('config'), f"{model_name}Config")
        except AttributeError:
            print(f"{model_name} not included!")
            exit()

        self.model = Model(self.config).to(device)
        self.__loadModel(checkpoint_dir)

    def __loadModel(self, checkpoint_dir):
        checkpoint_path = latest_checkpoint(checkpoint_dir)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def retrain(self, new_behaviors):
        dataset = BaseDataset('', self.news_path, behaviors_mode='data', behaviors_pd=new_behaviors)

        train_content_rec_model(self.model, dataset)

        self.model.eval()

    def evaluate(self, test_behaviors):
        auc, acc, macro_f1, micro_f1 = test_content_rec_model(self.model, test_behaviors)

        return auc, acc, macro_f1, micro_f1

    def getNews2Emb(self):
        news_dataloader = DataLoader(self.news_dataset,
                                     batch_size=self.config.batch_size, # * 16,
                                     shuffle=False,
                                     num_workers=self.config.num_workers,
                                     drop_last=False,
                                     pin_memory=False)

        news2vector = {}
        for minibatch in tqdm(news_dataloader,
                              desc="Calculating vectors for news"):
            news_ids = minibatch["id"]

            if any(id not in news2vector for id in news_ids):
                with torch.no_grad():
                    news_vector = self.model.get_news_vector(minibatch)
                    for id, vector in zip(news_ids, news_vector):
                        if id not in news2vector:
                            news2vector[id] = vector

        news2vector['PADDED_NEWS'] = torch.zeros(
            list(news2vector.values())[0].size()).to(device)

        return news2vector

    def getUser2Emb(self, news2vector=None):
        user_dataloader = DataLoader(self.user_dataset,
                                     batch_size=self.config.batch_size, # * 16,
                                     shuffle=False,
                                     num_workers=self.config.num_workers,
                                     drop_last=False,
                                     pin_memory=False)

        user2vector = {}
        if not news2vector:
            news2vector = self.getNews2Emb()
        for minibatch in tqdm(user_dataloader,
                              desc="Calculating vectors for users"):

            user_ids = minibatch["user"]

            if any(uid not in user2vector for uid in user_ids):
                clicked_news_vector = torch.stack([
                    torch.stack([news2vector[x].to(device) for x in news_list],
                                dim=0) for news_list in minibatch["clicked_news"]
                ],
                    dim=0).transpose(0, 1)

                # batch_size x num_clicked_news_a_user x user embeddding dim
                # print(clicked_news_vector.shape)

                if model_name == 'LSTUR':
                    with torch.no_grad():
                        user_vector = self.model.get_user_vector(
                            minibatch['id'], minibatch['clicked_news_length'],
                            clicked_news_vector)
                else:
                    with torch.no_grad():
                        user_vector = self.model.get_user_vector(clicked_news_vector)

                for user, vector in zip(user_ids, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector

        return user2vector

    def getChickProb(self, news_vector, user_vector):

        with torch.no_grad():
            if model_name != 'DKN':
                click_probability = self.model.click_predictor(news_vector, user_vector)
            else:
                clicked_news_vector = user_vector
                # batch_size, candidate_size, word_embedding_dim
                user_vector = torch.stack([
                    self.model.attention(x, clicked_news_vector)
                    for x in news_vector.transpose(0, 1)
                ],
                    dim=1)

                size = news_vector.size()
                click_probability = self.model.click_predictor(
                    news_vector.view(size[0] * size[1], size[2]),
                    user_vector.view(size[0] * size[1], size[2])).view(size[0], size[1])

        sigmod = nn.Sigmoid()
        click_probability = sigmod(click_probability)
        return click_probability


    def reloadUserDataset(self, behaviors_path, user2int_path):

        self.user_dataset = UserDataset(behaviors_path, user2int_path)


    def updateUserDataset(self, user2added_news):

        self.user_dataset.updateDataset(user2added_news)
