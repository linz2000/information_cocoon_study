from config import dataset_name, device
from config import NGCFConfig
from model.NGCF.ngcf import NGCF
from model.NGCF.parser import parse_args
from dataset import NGCF_Dataset

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import warnings
import os
import math
import pandas as pd
import pickle as pkl
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings('ignore')

args = parse_args()
args.device = device
args.batch_size = NGCFConfig.batch_size
args.epoch = NGCFConfig.epoch
args.weights_path = f'checkpoint/{dataset_name}/NGCF'
args.node_dropout = eval(args.node_dropout)
args.mess_dropout = eval(args.mess_dropout)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_generator, criterion, device):
    model.train()

    n_batch = data_generator.n_train // args.batch_size + 1
    for idx in tqdm(range(n_batch), desc='Training'):
        users, pos_items, neg_items = data_generator.sample()

        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                       pos_items,
                                                                       neg_items,
                                                                       drop_flag=args.node_dropout_flag)
        logits = model.get_logits(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        size = len(users)
        labels = torch.FloatTensor([1. for i in range(size)] +
                                   [0. for i in range(size)]).to(device)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def retrain(model, optimizer, added_data, criterion, args):
    model.train()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    added_users, added_pos_items, added_neg_items = added_data
    batch_size = args.batch_size
    n_batch = math.ceil( len(added_users) / batch_size )

    for idx in tqdm(range(n_batch), desc='Retraining'):
        users = added_users[idx * batch_size: (idx + 1) * batch_size]
        pos_items = added_pos_items[idx * batch_size: (idx + 1) * batch_size]
        neg_items = added_neg_items[idx * batch_size: (idx + 1) * batch_size]

        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                       pos_items,
                                                                       neg_items,
                                                                       drop_flag=args.node_dropout_flag)
        logits = model.get_logits(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        size = len(users)
        labels = torch.FloatTensor([1. for i in range(size)] +
                                   [0. for i in range(size)]).to(args.device)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, data_generator, mode):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        if mode == 'val':
            total_users, total_pos_items, total_neg_items = data_generator.get_val_data()
        else:
            total_users, total_pos_items, total_neg_items = data_generator.get_test_data()
        batch_size = args.batch_size
        n_batch = math.ceil(len(total_pos_items) / batch_size)

        for idx in tqdm(range(n_batch), desc='Testing'):
            users = total_users[idx * batch_size: (idx + 1) * batch_size]
            pos_items = total_pos_items[idx * batch_size: (idx + 1) * batch_size]
            neg_items = total_neg_items[idx * batch_size: (idx + 1) * batch_size]

            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)
            logits = model.get_logits(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            sigmod = nn.Sigmoid()
            preds = sigmod(logits)
            preds = preds.tolist()

            size = len(users)
            labels = [1 for i in range(size)] + [0 for i in range(size)]

            targets.extend(labels)
            predicts.extend(preds)

    auc = roc_auc_score(targets, predicts)

    return auc

def test_ngcf_model(model, data_generator):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        total_users, total_pos_items, total_neg_items = data_generator.get_test_data()
        batch_size = args.batch_size
        n_batch = math.ceil(len(total_pos_items) / batch_size)

        for idx in tqdm(range(n_batch), desc='Testing'):
            users = total_users[idx * batch_size: (idx + 1) * batch_size]
            pos_items = total_pos_items[idx * batch_size: (idx + 1) * batch_size]
            neg_items = total_neg_items[idx * batch_size: (idx + 1) * batch_size]

            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)
            logits = model.get_logits(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            sigmod = nn.Sigmoid()
            preds = sigmod(logits)
            preds = preds.tolist()

            size = len(users)
            labels = [1 for i in range(size)] + [0 for i in range(size)]

            targets.extend(labels)
            predicts.extend(preds)

    predicts_l = [1 if s > 0.5 else 0 for s in predicts]

    auc = roc_auc_score(targets, predicts)
    acc = accuracy_score(targets, predicts_l)
    macro_f1 = f1_score(targets, predicts_l, average='macro')
    micro_f1 = f1_score(targets, predicts_l, average='micro')

    return auc , acc, macro_f1, micro_f1

def get_news_int2emb(news2emb_path, news2int_path):

    news2int = dict(pd.read_table(news2int_path).values.tolist())
    with open(news2emb_path, 'rb') as embf:
        news2emb_arr = pkl.load(embf)

    news_int2emb = {}
    for news, emb_arr in news2emb_arr.items():
        if news in news2int:
            news_int2emb[news2int[news] - 1] = torch.from_numpy(emb_arr)

    return news_int2emb


def main():
    use_emb = False
    news2emb_path = f"data/{dataset_name}/train/news2emb.pkl"
    news2int_path = f"data/{dataset_name}/train/news2int.tsv"

    train_dir = f"data/{dataset_name}/train"
    train_data_path = f"data/{dataset_name}/train/ngcf_behaviors.txt"
    val_data_path = f"data/{dataset_name}/val/ngcf_behaviors.txt"
    test_data_path = f"data/{dataset_name}/test/ngcf_behaviors.txt"

    data_generator = NGCF_Dataset(train_dir, train_data_path, val_data_path, test_data_path, batch_size=args.batch_size)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()  # norm = mean + eye

    news_tensors = None
    if use_emb:
        news_int2emb = get_news_int2emb(news2emb_path, news2int_path)
        n_news = data_generator.n_items
        news_tensors = torch.stack([news_int2emb[i] for i in range(n_news)], dim=0)
        args.embed_size = 300 # embedding dim


    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args,
                 news_tensors=news_tensors).to(args.device)

    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)
    model_savepath = os.path.join(args.weights_path, 'ckpt.pth')
    if os.path.exists(model_savepath):
        model.load_state_dict(torch.load(model_savepath, map_location=args.device))
        model.eval()
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        early_stopper = EarlyStopper(num_trials=5, save_path=model_savepath)

        for epoch_i in range(args.epoch):
            train(model, optimizer, data_generator, criterion, args.device)
            auc = test(model, data_generator, mode='val')
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break

    auc = test(model, data_generator, mode='test')
    print(f'final test auc: {auc}')

if __name__ == '__main__':
    main()