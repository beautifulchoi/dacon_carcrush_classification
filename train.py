import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from config import CFG
from utils import *
import warnings
warnings.filterwarnings(action='ignore')


def train_one_epoch(model, criterion, optimizer, train_loader, device):
    train_loss = []
    model.train()
    for videos, labels in tqdm(iter(train_loader)):
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(videos)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    return train_loss  # score 변화없으면 잘못된거니까 바꾸기 -> 업데이트를 안하는거겠지 옵티마이저 반환안해줘서


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            # if model.module.binary==True: #because of DataParallel
            #    labels=labels.float()

            logit = model(videos)

            # loss = criterion(logit, labels.reshape(-1,1))
            loss = criterion(logit, labels)
            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()

            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score


def train(model, criterion, optimizer, train_loader, val_loader, scheduler, device, is_parallel=False):
    model.to(device)

    if torch.cuda.device_count() > 1:
        if is_parallel == True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    criterion = criterion().to(device)
    best_val_score = 0
    best_model = None
    cnt = 0

    for epoch in range(1, CFG.epochs+1):
        train_loss = train_one_epoch(
            model, criterion, optimizer, train_loader, device)  # train
        _val_loss, _val_score = validation(
            model, criterion, val_loader, device)  # validation
        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

            cnt = 0
        else:
            print("early stopping count : {}".format(cnt+1))
            cnt += 1

        if best_val_score >= 0.999:
            print("already on best score")
            break

        if cnt == CFG.earlystop:
            print("early stopping done")
            break

    return best_model


def run(model, df, name: str, transforms, device, save_dir, is_fold=True, is_aug=False):
    if is_fold != True:
        loop = 1
    else:
        loop = CFG.fold

    for k in range(loop):
        # data load for each fold
        if is_aug == True:
            _, _, train_loader, val_loader = load_dataset(
                df, name, k, transforms=transforms, make_data=True)
        else:
            _, _, train_loader, val_loader = load_dataset(
                df, name, k, transforms=transforms)

        model.eval()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.00001)
        criterion = nn.CrossEntropyLoss  # 추후 수정 예정

        print("{} model run".format(name))
        print("{}th model run".format(k+1))
        print("_"*100)

        infer_model = train(model, criterion, optimizer,
                            train_loader, val_loader, scheduler, device)

        os.makedirs(save_dir, exist_ok=True)
        if is_fold == True:
            torch.save(infer_model.state_dict(
            ), save_dir+'/{}_{}fold.pt'.format(name, k))
        else:
            torch.save(infer_model.state_dict(
            ), save_dir+'/{}.pt'.format(name))

        del infer_model
