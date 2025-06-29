import argparse
import collections
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (auc,recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score,
                             accuracy_score, precision_recall_curve, auc, mean_absolute_error)
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from parse_config import ConfigParser
from trainer import GraghTrainer, CombinedTrainer
from utils import prepare_device

import data_loader as module_data
import model as module_model
import loss as module_loss
import metric as module_metric

from model.composition import Roost_CrabNet, Roost_CrabNet_reg
from data_loader import CombinedDataLoader, get_combined_dataset, custom_collate_fn
from predict_test import get_model_output

def loader():

    # data = np.load(path, allow_pickle=True)
    args = {
            "roost_data_dir": "data/MP/mp_data_Roost.csv",
            "crabnet_data_dir" : "data/MP/mp_data_CrabNet.csv",
            "magpie_data_dir" : "dataset/magpie_fea.npy",
            "meredig_data_dir": "dataset/meredig_fea.npy",
            "rsc_data_dir" : "dataset/RSC_fea.npy",
            "batch_size": 128,
            "shuffle": False,
            "use_Kfold": True,
            "total_fold":5,
            "nth_fold":0
        }
    # data_loader = module_data.GraphDataLoader(**args)
    #
    # valid_data_loader = data_loader.split_validation()
    # test_data_loader = data_loader.split_test()
    data_loader, valid_data_loader, test_data_loader = CombinedDataLoader(**args)._split_sampler()


    return valid_data_loader, test_data_loader


def predict_combined(model_path, model_args, data_loader, device='cuda:0'):


    state_dict = torch.load(model_path)['state_dict']
    model = Roost_CrabNet(**model_args).to(device)
    model.load_state_dict(state_dict )

    predict_values = []
    target_values = []
    with torch.no_grad():
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                        roost_targets,
                        roost_ids,
                        crabnet_src,
                        crabnet_targets,
                        crabnet_ids,
                        magpie_fea,
                        meredig_fea,
                        rsc_dea) in tqdm(enumerate(data_loader)):
            roost_elem_weights = roost_elem_weights.to(device)
            roost_elem_fea = roost_elem_fea.to(device)
            roost_self_idx = roost_self_idx.to(device)
            roost_nbr_idx = roost_nbr_idx.to(device)
            roost_crystal_elem_idx = roost_crystal_elem_idx.to(device)

            crabnet_src = crabnet_src.to(device)

            src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
            frac = frac * (1 + (torch.randn_like(frac)) * 0.02)
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
            src = src.long()
            frac = frac.float()

            src, frac = src.to(device), frac.to(device)
            targets = crabnet_targets.to(device).view(-1, 1)

            magpie_fea = magpie_fea.to(device)
            meredig_fea = meredig_fea.to(device)
            rsc_dea = rsc_dea.to(device)

            y = model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx,
                                roost_crystal_elem_idx, src, frac, magpie_fea, meredig_fea, rsc_dea)



            predict_values.append(y.detach().cpu().numpy())
            target_values.append(targets.detach().cpu().numpy())

            torch.cuda.empty_cache()

        y = np.vstack(predict_values)
        target = np.vstack(target_values)


        # y = torch.cat(predict_values, 0)
        # y = y.cpu()
        # y = y.detach().numpy()

    return y, target

def y_to_01(y):
    new_y = []
    for i in range(len(y)):
        if y[i] > 0.5:
            new_y.append(1)
        else:
            new_y.append(0)
    return np.array(new_y)

def Performance( pre_test_y_prob, test_y):

    test_y = test_y.astype(int)
    pre_test_y = y_to_01(pre_test_y_prob)
    accuracy = accuracy_score(test_y, pre_test_y)

    precision, recall, _ = precision_recall_curve(test_y, pre_test_y_prob)
    aupr = auc(recall, precision)
    max_f1 = max(2 * (precision * recall) / (precision + recall))


    precision = precision_score(test_y, pre_test_y, zero_division=0)
    recall = recall_score(test_y, pre_test_y)
    f1 = f1_score(test_y, pre_test_y)
    fnr = confusion_matrix(test_y, pre_test_y, normalize='pred')[1][0]
    auc_score = roc_auc_score(test_y, pre_test_y_prob)

    tn, fp, fn, tp = confusion_matrix(test_y, pre_test_y).ravel()
    npv = tn / (tn + fn)


    return accuracy,precision,recall,f1,npv,auc_score, aupr,max_f1





if __name__ == '__main__':
    data_param = {
        "roost_data_path": "data/Li_S/roost_Li_S.csv",
        "crabnet_data_path": "data/Li_S/crabnet_Li_S.csv",
        "magpie_data_path": "data/Li_S/magpie_f_Li_S.npy",
        "meredig_data_path": "data/Li_S/meredig_f_Li_S.npy",
        "rsc_data_path": "data/Li_S/Li_S_RSC_f.npy",
        "ec_data_path": "data/Li_S/Li_S_EC_f.npy"
    }
    # roost_data_path, crabnet_data_path, magpie_data_path, meredig_data_path, rsc_data_path, ec_data_path
    model_param = {
        "elem_emb_len": 200,
        "elem_fea_len": 512,
        "n_graph": 3,
        "elem_heads": 4,
        "cry_heads": 4,
        "d_model": 512,
        "crab_n": 3,
        "crab_heads": 4
    }

    model = Roost_CrabNet_reg(**model_param)
    dataset = get_combined_dataset(**data_param)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)

    checkpoint_path = r"saved/models/Roost_Crab/0113_114456/model_best.pth"
    # acc, auc = test_swa_model(model, test_loader, device, checkpoint_path, 82, 92, optimizer, torch.nn.BCELoss())

    train_X = []
    val_X = []
    test_X = []

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # swa_model = get_swa_model(model, checkpoint_path, 45, 50)
    pre_y, target_y = get_model_output(model, loader, device)
    # pre_y = test_out.cpu().numpy()
    # target_y = get_targets(test_loader)
    mae = mean_absolute_error(target_y, pre_y)
    print(mae)

    # for epoch in range(100, 110):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(1.65, 1.65))
    plt.rcParams['font.family'] = 'Arial'
    plt.scatter(pre_y, target_y, s=1)
    plt.xlabel('Predicted Values', fontsize=8)
    plt.ylabel('Target Values', fontsize=8)
    # plt.title('Scatter Plot', fontsize=8)
    plt.xlim(-4.5, 1.3)
    plt.ylim(-4.5, 1.3)
    plt.plot(np.arange(-4.5, 1.4, 0.1), np.arange(-4.5, 1.4, 0.1))
    plt.xticks(np.arange(-4, 2, 1), fontsize=5)
    plt.yticks(np.arange(-4, 2, 1), fontsize=5)

    plt.text(-4.2, 0.3, f'MAE: {mae:.4f}', fontsize=6, ha='left')
    plt.show()

    np.save('data/Li_S/Li_S_combine_429.npy', pre_y)
    np.save('data/Li_S/Li_S_target_429.npy', target_y)
