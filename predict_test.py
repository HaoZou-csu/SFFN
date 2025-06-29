import torch
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np

from model.composition import Roost_CrabNet, Roost_CrabNet_reg, Single_feature_net, CrabNet, Roost, Simple_cat_reg, Transformer_reg, Gated_fusion_network
from data_loader import CombinedDataLoader, get_combined_dataset, custom_collate_fn

from utils.util import Scaler

def get_model_output(model, test_loader, device):
    model = model.to(device)
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                        roost_targets,
                        roost_ids,
                        crabnet_src,
                        crabnet_targets,
                        crabnet_ids,
                        magpie_fea,
                        meredig_fea,
                        rsc_fea,
                        ec_fea) in enumerate(test_loader):
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
            magpie_fea = magpie_fea.to(device)
            meredig_fea = meredig_fea.to(device)
            rsc_fea = rsc_fea.to(device)
            ec_fea = ec_fea.to(device)

            # Forward pass to get the intermediate output

            batch_out = model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx, src, frac, magpie_fea, meredig_fea, rsc_fea, ec_fea)
            target = crabnet_targets.view(-1, 1)

            outputs.append(batch_out)
            targets.append(target.cpu().numpy())

        outputs = torch.cat(outputs, dim=0).cpu().numpy()
        targets = np.concatenate(targets)
    return outputs, targets

def get_single_output(model, test_loader, device):
    model = model.to(device)
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                        roost_targets,
                        roost_ids,
                        crabnet_src,
                        crabnet_targets,
                        crabnet_ids,
                        magpie_fea,
                        meredig_fea,
                        rsc_fea,
                        ec_fea) in enumerate(test_loader):
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
            magpie_fea = magpie_fea.to(device)
            meredig_fea = meredig_fea.to(device)
            rsc_fea = rsc_fea.to(device)
            ec_fea = ec_fea.to(device)

            # Forward pass to get the intermediate output

            batch_out = model(magpie_fea)
            target = crabnet_targets.view(-1, 1)

            outputs.append(batch_out)
            targets.append(target.cpu().numpy())

        outputs = torch.cat(outputs, dim=0).cpu().numpy()
        targets = np.concatenate(targets)
    return outputs, targets

def get_roost_output(model, test_loader, device):
    model = model.to(device)
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                        roost_targets,
                        roost_ids,
                        crabnet_src,
                        crabnet_targets,
                        crabnet_ids,
                        magpie_fea,
                        meredig_fea,
                        rsc_fea,
                        ec_fea) in enumerate(test_loader):
            roost_elem_weights = roost_elem_weights.to(device)
            roost_elem_fea = roost_elem_fea.to(device)
            roost_self_idx = roost_self_idx.to(device)
            roost_nbr_idx = roost_nbr_idx.to(device)
            roost_crystal_elem_idx = roost_crystal_elem_idx.to(device)

            crabnet_src = crabnet_src.to(device)

            target = crabnet_targets.view(-1, 1)

            magpie_fea = magpie_fea.to(device)
            meredig_fea = meredig_fea.to(device)
            rsc_fea = rsc_fea.to(device)
            ec_fea = ec_fea.to(device)

            output = model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx,
                                roost_crystal_elem_idx)
            outputs.append(output[0])
            targets.append(target.cpu().numpy())

        outputs = torch.cat(outputs, dim=0).cpu().numpy()
        targets = np.concatenate(targets)
    return outputs, targets


def get_crabnet_output(model, test_loader, device):
    model = model.to(device)
    model.eval()
    outputs = []
    targets = []

    train_y = np.load('data/MP/y_train.npy')
    scaler = Scaler(train_y)

    with torch.no_grad():
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                        roost_targets,
                        roost_ids,
                        crabnet_src,
                        crabnet_targets,
                        crabnet_ids,
                        magpie_fea,
                        meredig_fea,
                        rsc_fea,
                        ec_fea) in enumerate(test_loader):
            crabnet_src = crabnet_src.to(device)

            src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
            frac = frac * (1 + (torch.randn_like(frac)) * 0.02)  # normal
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
            src = src.long()
            frac = frac.float()

            src, frac = src.to(device), frac.to(device)
            target = crabnet_targets.to(device).view(-1, 1)

            output = model(src, frac)

            # scaler = Scaler(target)

            output, uncertainty = output.chunk(2, dim=-1)
            uncertainty = torch.exp(uncertainty) * scaler.std
            output = scaler.unscale(output)

            outputs.append(output)
            targets.append(target.cpu().numpy())

        outputs = torch.cat(outputs, dim=0).cpu().numpy()
        targets = np.concatenate(targets)
    return outputs, targets




if __name__ == '__main__':
    data_param = {
        "roost_data_dir": "data/MP/mp_roost_hf.csv",
        "crabnet_data_dir": "data/MP/mp_crabnet_hf.csv",
        "magpie_data_dir": "dataset/magpie_fea.npy",
        "meredig_data_dir": "dataset/meredig_fea.npy",
        "rsc_data_dir": "dataset/RSC_fea.npy",
        "ec_data_dir": "dataset/EC_fea.npy",
        "batch_size": 128,
        "shuffle": False,
        "random_seed": 42,
        "use_Kfold": True,
        "total_fold": 5,
        "nth_fold": 0,
        "load_from_local_file": "data/MP/mp_prediction_split.pkl"
    }

    test_data_param = {
        "roost_data_path": "data/candidates/single_perovskite_halide/roost_single_perovskite_halide.csv",
        "crabnet_data_path": "data/candidates/single_perovskite_halide/crabnet_single_perovskite_halide.csv",
        "magpie_data_path": "data/candidates/single_perovskite_halide/magpie_single_perovskite_halide.npy",
        "meredig_data_path": "data/candidates/single_perovskite_halide/meredig_single_perovskite_halide.npy",
        "rsc_data_path": "data/candidates/single_perovskite_halide/rsc_single_perovskite_halide.npy",
        "ec_data_path": "data/candidates/single_perovskite_halide/ec_single_perovskite_halide.npy"
    }


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
    # roost_data_path, crabnet_data_path, magpie_data_path, meredig_data_path, rsc_data_path, ec_data_path
    test_dataset = get_combined_dataset(**test_data_param)

    model = Roost_CrabNet_reg(**model_param)
    loader = CombinedDataLoader(**data_param)
    train_loader, val_loader, test_loader = loader._split_sampler()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = r"saved/models/Roost_Crab_prediction/0115_100610/model_best.pth"
    # acc, auc = test_swa_model(model, test_loader, device, checkpoint_path, 82, 92, optimizer, torch.nn.BCELoss())


    train_X = []
    val_X = []
    test_X = []

    epoch = 100
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # swa_model = get_swa_model(model, checkpoint_path, 45, 50)
    pre_y, target_y = get_model_output(model, test_loader, device)
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
    plt.xticks(np.arange(-4,2,1),fontsize=5)
    plt.yticks(np.arange(-4,2,1),fontsize=5)

    plt.text(-4.2, 0.3, f'MAE: {mae:.4f}', fontsize=6, ha='left')

    plt.show(block=True)

    np.save('data/candidates/single_perovskite_halide/prediction_single_perovskite_halide.npy', pre_y)
    # np.save('data/Li_S/Li_S_combine_429.npy', pre_y)
    # np.save('data/Li_S/Li_S_target_429.npy', target_y)


