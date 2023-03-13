# from balanced_loss import Loss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from config import CFG
from utils import *
import warnings
from copy import deepcopy
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

    return train_loss


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


def train(model, criterion, optimizer, train_loader, val_loader, scheduler, device, writer, is_parallel=False, ce_weight=None):
    model.to(device)

    if torch.cuda.device_count() > 1:
        if is_parallel == True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    criterion = criterion().to(device)
    if ce_weight != None:
        ce_weight = ce_weight.to(device)
        # print(next(model.parameters()).is_cuda)  # 모델 들어있나 체크
    best_val_score = 0
    best_model = None
    cnt = 0

    for epoch in range(1, CFG.epochs+1):
        if ce_weight != None:
            train_loss = train_one_epoch(
                model, ce_weight, optimizer, train_loader, device)  # train
        else:
            train_loss = train_one_epoch(
                model, criterion, optimizer, train_loader, device)  # train

        _val_loss, _val_score = validation(
            model, criterion, val_loader, device)  # validation
        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

        # write loss of each epoch
        writer.add_scalar('Train/Loss', _train_loss.item(), epoch)
        writer.add_scalar('Val/Loss', _val_loss.item(), epoch)
        writer.add_scalar('Val/F1 Score', _val_score, epoch)

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
            writer.close()
            break

        if cnt == CFG.earlystop:
            print("early stopping done")
            writer.close()
            break

    return best_model

# 기존 모델도 삭제하려면-> model 선언 시 객체 자체를 넘긴다면?


def run(Model: nn.Module, df, name: str, transforms, device, save_dir, is_fold=True, is_aug=False, is_parallel=False, weight=None, weighted_sampling=None):
    if is_fold != True:
        loop = 1
    else:
        loop = CFG.fold

    for k in range(loop):

        # tensorboard 추가
        writer = SummaryWriter(save_dir+f'/{k}fold')

        # model reload for each step
        if (name == "weather") or (name == "crush+ego"):
            model = Model(num_classes=3, binary=False)
        else:
            model = Model(num_classes=2)

        # data load for each fold
        if is_aug == True:
            _, _, train_loader, val_loader = load_dataset(
                df, name, k, transforms=transforms, make_data=True, weighted_sampling=weighted_sampling)
        else:
            _, _, train_loader, val_loader = load_dataset(
                df, name, k, transforms=transforms, weighted_sampling=weighted_sampling)

        model.eval()
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=CFG.lr)  # 변경
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.00001)
        criterion = nn.CrossEntropyLoss  # 추후 수정 예정
        # criterion = FocalLoss

        if weight != None:
            ce_weight = criterion(weight=weight)
        else:
            ce_weight = None

        print("{} model run".format(name))
        print("{}th model run".format(k+1))
        print("_"*100)

        infer_model = train(model, criterion, optimizer,
                            train_loader, val_loader, scheduler, device, writer, is_parallel, ce_weight=ce_weight)

        os.makedirs(save_dir, exist_ok=True)
        if is_fold == True:
            torch.save(infer_model.state_dict(
            ), save_dir+'/{}_{}fold.pt'.format(name, k))
        else:
            torch.save(infer_model.state_dict(
            ), save_dir+'/{}.pt'.format(name))

        del infer_model
        del model


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction=self.reduction)
        pt = torch.exp(-ce_loss)

        focal_loss = ((1-pt)**self.gamma * ce_loss).mean()

        return focal_loss
