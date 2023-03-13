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


def run_infer(Model: torch.nn.Module, name: str, path: str, device, test_loader, fold=True, is_parallel=False):
    preds_each = []

    for k in range(CFG.fold):
        # model reload for each step
        if (name == "weather") or (name == "mix"):
            model = Model(num_classes=3, binary=False)
        else:
            model = Model(num_classes=2)

        if fold == True:
            re_path = path[:-3]+'_{}fold.pt'.format(k)
        else:
            re_path = path
        state_dict = torch.load(re_path)

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

    del model

    return preds_each


def voting(name, preds: list, save_each: str = None, save_result: str = None):
    vote_preds = []

    for k in range(CFG.fold):
        sample = pd.read_csv('./sample_submission.csv')

        sample[name] = preds[k]
        if save_each != None:
            sample.to_csv(save_each+f'{name}_fold{k}.csv', index=False)

    cols = list(zip(*preds))
    for c in cols:
        most = Counter(c).most_common()[0][0]
        vote_preds.append(most)

    ss = pd.read_csv('./sample_submission.csv')
    ss[name] = vote_preds
    if save_result != None:
        ss.to_csv(save_result+'vote_{}.csv'.format(name), index=False)

    return vote_preds, ss

def vote_result(ss_list:list , save_dir: str = None):
#ss_list에 csv 경로 담기
    preds=[]
    vote_preds=[]
    result = pd.read_csv('./sample_submission.csv')
    for ss in ss_list: 
        csv=pd.read_csv(ss)
        preds.append(csv['label'].tolist())
    cols = list(zip(*preds))
    for c in cols:
        most = Counter(c).most_common()[0][0]
        vote_preds.append(most)

    result['label'] = vote_preds
    if save_dir != None:
        result.to_csv(save_dir+'ensemble_models.csv', index=False)

    return result,cols
