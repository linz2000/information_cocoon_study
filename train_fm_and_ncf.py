from config import model_name, dataset_name, device
from dataset import FMDataset
from config import FMConfig, MINDConfig

import torch
from torch.utils.data import DataLoader
import importlib
from pathlib import Path
import os
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

def get_model(field_dims):
    if model_name[-2:] == 'ID':
        Model = getattr(importlib.import_module(f"model.FM.{model_name[:-3].lower()}"), model_name[:-3])
    else:
        Model = getattr(importlib.import_module(f"model.FM.{model_name.lower()}"), model_name)
    if model_name == 'DFM' or model_name == 'DFM_ID':
        return Model(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif model_name == 'NFM' or model_name == 'NFM_ID':
        return Model(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif model_name == 'NCF':
        return Model(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                     user_field_idx=np.array(MINDConfig.user_idx, dtype=np.long),
                                     item_field_idx=np.array(MINDConfig.item_idx, dtype=np.long))
    else:
        raise ValueError('unknown model name: ' + model_name)

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

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            # user_emb = model.get_embs(fields[:,:1], MINDConfig.user_idx)
            # item_emb = model.get_embs(fields[:,1:], MINDConfig.item_idx)

    return roc_auc_score(targets, predicts)

def test_fm_model(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    predicts_l = [1 if s>0.5 else 0 for s in predicts]

    auc = roc_auc_score(targets, predicts)
    acc = accuracy_score(targets, predicts_l)
    macro_f1 = f1_score(targets, predicts_l, average='macro')
    micro_f1 = f1_score(targets, predicts_l, average='micro')

    return auc, acc, macro_f1, micro_f1

def main():
    train_behaviors_path = "data/MIND/train/fm_behaviors.tsv"
    val_behaviors_path = "data/MIND/val/fm_behaviors.tsv"
    test_behaviors_path = "data/MIND/test/fm_behaviors.tsv"
    model_savepath = f"checkpoint/{dataset_name}/{model_name}/ckpt.pth"

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists(f'checkpoint/{dataset_name}'):
        os.makedirs(f'checkpoint/{dataset_name}')
    checkpoint_dir = os.path.join(f'./checkpoint/{dataset_name}', model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if model_name == 'NCF' or model_name == 'DFM_ID' or model_name =='NFM_ID':
        train_dataset = FMDataset.init_from_path(train_behaviors_path, id_only=True)
        val_dataset = FMDataset.init_from_path(val_behaviors_path, id_only=True)
        test_dataset = FMDataset.init_from_path(test_behaviors_path, id_only=True)
    else:
        train_dataset = FMDataset.init_from_path(train_behaviors_path)
        val_dataset = FMDataset.init_from_path(val_behaviors_path)
        test_dataset = FMDataset.init_from_path(test_behaviors_path)

    train_data_loader = DataLoader(train_dataset, batch_size=FMConfig.batch_size, num_workers=8)
    val_data_loader = DataLoader(val_dataset, batch_size=FMConfig.batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=FMConfig.batch_size, num_workers=8)

    if model_name == 'NCF' or model_name == 'DFM_ID' or model_name =='NFM_ID':
        model = get_model(np.array(MINDConfig.id_field_dims)).to(device)
    else:
        model = get_model(np.array(MINDConfig.field_dims)).to(device)

    if os.path.exists(model_savepath):
        model.load_state_dict(torch.load(model_savepath, map_location=device))
        model.eval()
    else:
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=FMConfig.learning_rate, weight_decay=FMConfig.weight_decay)
        early_stopper = EarlyStopper(num_trials=5, save_path=model_savepath)
        for epoch_i in range(FMConfig.epoch):
            train(model, optimizer, train_data_loader, criterion, device)
            auc = test(model, val_data_loader, device)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break

    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    main()