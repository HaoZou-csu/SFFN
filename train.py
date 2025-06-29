import argparse
import collections
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
import os

from parse_config import ConfigParser
from trainer import GraghTrainer, CombinedTrainer, CrabnetTrainer, SingleTrainer, RoostTrainer
from utils import prepare_device

import data_loader as module_data

import model as module_model

import loss as module_loss
import metric as module_metric

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(config):
    logger = config.get_logger('train')
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    data_loader, valid_data_loader, test_loader = data_loader._split_sampler()
    # build model architecture, then print to console
    model = config.init_obj('arch', module_model)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    print(device)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    ## 载入预训练权重
    # n_targets = [132]  # 根据任务字典设置目标数量
    # elem_emb_len = 200
    # m_model = module_model.m_Roost(n_targets=n_targets,
    # elem_emb_len=elem_emb_len,
    # elem_fea_len=64,
    # n_graph=3,
    # elem_heads=3,
    # elem_gate=(256,),
    # elem_msg=(256,),
    # cry_heads=3,
    # cry_gate=(256,),
    # cry_msg=(256,),
    # trunk_hidden=[1024, 512],
    # out_hidden=[256, 128, 64])
    # best_model_path = r'D:\programming\pretrain_roost\best_model.pth'
    # m_model.load_state_dict(torch.load(best_model_path))
    # m_model.eval()
    # m_model_state_dict = m_model.state_dict()
    # material_nn_weights = {k: v for k, v in m_model_state_dict.items() if k.startswith('material_nn.')}
    # model_state_dict = model.state_dict()
    # model_state_dict.update(material_nn_weights)
    # model.load_state_dict(model_state_dict)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config['model_type'] == 'Graph':
        trainer = GraghTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

    elif config['model_type'] == 'Combine':
        trainer = CombinedTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)
    elif config['model_type'] == 'Crabnet':
        trainer = CrabnetTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          )
    elif config['model_type'] == 'Single':
        trainer = SingleTrainer(model, criterion, metrics, optimizer,
                         config=config,
                         device=device,
                         data_loader=data_loader,
                         valid_data_loader=valid_data_loader,
                         lr_scheduler=lr_scheduler)
    elif config['model_type'] == 'Roost':
        trainer = RoostTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

    trainer.train()
    acc, auc = trainer.test(test_loader)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='CGCNN temp')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)



