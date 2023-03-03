from data import CustomDataset, Transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from config import CFG
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from matplotlib import pyplot as plt
import random


# get data loader


def load_dataset(df, name, k, transforms=Transforms.other, get_loader=True, make_data=False, weighted_sampling=None):

    train_df = df[df["fold"] != k]
    val_df = df[df["fold"] == k]

    train_dataset = CustomDataset(
        train_df['video_path'].values, train_df[name].values, transforms=transforms)
    val_dataset = CustomDataset(
        val_df['video_path'].values, val_df[name].values, transforms=Transforms.test)

    if make_data == True:
        if name == "weather":
            for tr_fold in range(CFG.fold):
                if tr_fold == k:  # 검증셋에서는 통과
                    print(f"{tr_fold}fold: 검증셋은 통과")
                    continue
                aug_dset_rain = make_cross_augment(
                    df, 2, tr_fold, sampling=100)
                aug_dset_snow = make_cross_augment(
                    df, 1, tr_fold, sampling=120)
                train_dataset = ConcatDataset(
                    [train_dataset, aug_dset_rain, aug_dset_snow])

        elif name == "timing":
            for tr_fold in range(CFG.fold):
                if tr_fold == k:  # 검증셋에서는 통과
                    print(f"{tr_fold}fold: 검증셋은 통과")
                    continue
                aug_dset_night = make_cross_augment(
                    df, 1, tr_fold, sampling=50, type=name)

                train_dataset = ConcatDataset(
                    [train_dataset, aug_dset_night])

    if get_loader == True:
        if weighted_sampling != None:
            _weights = cal_weight(df, name)
            train_dataset = train_dataset
            weighted_sampling = np.array(
                [_weights[t] for _, t in train_dataset])  # 너무 오래걸리는디
            weighted_sampling = torch.from_numpy(weighted_sampling)

            sampler = WeightedRandomSampler(
                weighted_sampling, num_samples=len(train_dataset), replacement=True)

            train_loader = DataLoader(
                train_dataset, batch_size=CFG.batch_size, sampler=sampler, num_workers=2)
            val_loader = DataLoader(
                val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(
                val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

        return train_dataset, val_dataset, train_loader, val_loader

    return train_dataset, val_dataset


def split_weather_subset(df):

    df_rain = df[df['weather'] == 2]
    df_snow = df[df['weather'] == 1]
    df_normal = df[df['weather'] == 0]
    return df_normal, df_snow, df_rain


def split_timing_subset(df):

    df_day = df[df['timing'] == 0]
    df_night = df[df['timing'] == 1]
    return df_day, df_night


def weather_augment(df, k):
    df_normal, df_snow, df_rain = split_weather_subset(df)
    tr1, val1 = load_dataset(df_normal, 'weather', k, Transforms.other, False)
    tr2, val2 = load_dataset(df_snow, 'weather', k, Transforms.snow, False)
    tr3, val3 = load_dataset(df_rain, 'weather', k, Transforms.rain, False)

    tr = ConcatDataset([tr1, tr2, tr3])
    val = ConcatDataset([val1, val2, val3])

    train_loader = DataLoader(
        tr, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=CFG.batch_size,
                            shuffle=False, num_workers=2)

    return tr, val, train_loader, val_loader


# k번째 fold에 대한 augment 적용
def make_cross_augment(df, target_index: int, k: int, sampling=20, type='weather'):
    # 여기서 샘플링할 때 k를 고려해줘야한다는거
    if type == 'weather':
        df_normal, _, _ = split_weather_subset(df)
        sample_df = df_normal[df_normal['fold'] == k]
        sample_df = sample_df.sample(n=sampling, random_state=CFG.seed)
        sample_df = sample_df.reset_index(drop=True)
        sample_df['weather'] = target_index

        if target_index == 1:  # snow
            dset = CustomDataset(
                sample_df['video_path'].values, sample_df['weather'].values, transforms=Transforms.snow)
        elif target_index == 2:  # rain
            dset = CustomDataset(
                sample_df['video_path'].values, sample_df['weather'].values, transforms=Transforms.rain)
        else:
            ValueError("타겟인덱스 잘못정함: 1이나 2여야")

    elif type == 'timing':
        df_day, df_night = split_timing_subset(df)
        sample_df = df_day[df_day['fold'] == k]
        sample_df = sample_df.sample(n=sampling, random_state=CFG.seed)
        sample_df = sample_df.reset_index(drop=True)
        sample_df['timing'] = target_index

        if target_index == 1:  # dark
            dset = CustomDataset(
                sample_df['video_path'].values, sample_df['timing'].values, transforms=Transforms.darken)
        else:
            ValueError("타겟인덱스 잘못정함: 1이여야")
    else:
        ValueError("choose weather or timing")
    return dset


def cal_weight(df, name):
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(df[name]), y=df[name])
    weights = torch.tensor(class_weights, dtype=torch.float)

    return weights

# visualize transformed dataset


def visualize(dataset, idx=None, frame=0):
    if idx == None:
        idx = random.randint(0, len(dataset)-1)

    video, label = dataset[idx][0], dataset[idx][1]
    img = (video.permute(1, 2, 3, 0))[frame]
    plt.title(f"label: {label}")
    plt.imshow(img)
    plt.show()
    return img, label
