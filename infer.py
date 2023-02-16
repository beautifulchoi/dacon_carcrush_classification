import torch
from collections import OrderedDict, Counter
from tqdm.auto import tqdm
from utils import CFG
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)

            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

# path: ./weights/r3d_per_label_{}_Transform


def run_infer(model, name: str, path: str, device, test_loader, fold=True, is_parallel=False):
    preds_each = []

    for k in range(CFG.fold):
        if fold == True:
            path = path[:-3]+'_{}fold.pt'.format(k)
        state_dict = torch.load(path)

        if is_parallel == True:
            keys = state_dict.keys()
            values = state_dict.values()

            new_keys = []

            for key in keys:  # 병렬 처리 했을 경우에만
                new_key = key[7:]    # remove the 'module.'
                new_keys.append(new_key)

            new_dict = OrderedDict(list(zip(new_keys, values)))
            model.load_state_dict(new_dict)

        else:
            model.load_state_dict(state_dict)

        print(f"{path} key matching successfully")
        preds = inference(model, test_loader, device)
        preds_each.append(preds)

        if fold != True:
            break

    return preds_each


def voting(name, preds: list, save_each=False, save_result=False):
    vote_preds = []

    for k in range(CFG.fold):
        sample = pd.read_csv('./sample_submission.csv')

        sample[name] = preds[k]
        if save_each == True:
            sample.to_csv(f'./ensemble/weather_fold{k}.csv', index=False)

    cols = list(zip(*preds))
    for c in cols:
        most = Counter(c).most_common()[0][0]
        vote_preds.append(most)

    ss = pd.read_csv('./sample_submission.csv')
    ss[name] = vote_preds
    if save_result == True:
        ss.to_csv('vote_{}.csv'.format(name), index=False)

    return vote_preds, ss