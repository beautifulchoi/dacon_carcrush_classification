from data import CustomDataset, Transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from config import CFG

# get data loader


def load_dataset(df, name, k, transforms=Transforms.other, get_loader=True, make_data=False):

    train_df = df[df["fold"] != k]
    val_df = df[df["fold"] == k]

    train_dataset = CustomDataset(
        train_df['video_path'].values, train_df[name].values, transforms=transforms)
    val_dataset = CustomDataset(
        val_df['video_path'].values, val_df[name].values, transforms=Transforms.test)

    if make_data == True:  # just for weather
        for tr_fold in range(CFG.fold):
            if tr_fold == k:  # 검증셋에서는 통과
                print("검증셋은 통과")
                continue
            aug_dset_rain = make_cross_augment(df, 2, k, sampling=50)
            aug_dset_snow = make_cross_augment(df, 1, k, sampling=50)
            train_dataset = ConcatDataset(
                [train_dataset, aug_dset_rain, aug_dset_snow])

    if get_loader == True:
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
def make_cross_augment(df, target_index: int, k: int, sampling=20):
    # 여기서 샘플링할 때 k를 고려해줘야한다는거
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

    return dset
