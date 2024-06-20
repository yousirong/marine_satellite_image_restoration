import torch
import torch.nn as nn
import yaml
import shutil
import os
import re

def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def is_available_to_store(p=10):
    diskLabel = os.getcwd()
    total, used, free = shutil.disk_usage(diskLabel)
    div = 1000*3
    total, used, free = total/div, used/div, free/div
    free_space_percentile = int(free/total*100)
    return free_space_percentile > p


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def build_save_folder(args):
    if not args.test and not args.val:
        if args.model_path is None:
            assert not os.path.isdir(args.model_save_path), "Please check model_save_path, it already exists. >> "+os.path.abspath(args.model_save_path)
            os.makedirs(args.model_save_path)
        config_save_path= os.path.join(args.model_save_path, os.path.basename(args.c) )
        shutil.copyfile(args.c, config_save_path)

        if args.model_path is None:
            assert not os.path.isdir(f'./model/training/{os.path.basename(args.model_save_path)}')
            os.makedirs(f'./model/training/{os.path.basename(args.model_save_path)}')

    if args.test:
        directory, filename = os.path.split(args.model_path)
        model_dir = directory.split("/")[-1]  # Get the second last element from the split path
        _, loss_rate = os.path.split(args.mask_root)
        _,t_dataset = os.path.split(args.data_root)

        step = re.search(r'\d+', filename).group()
        args.result_save_path = os.path.join('./model/results', model_dir+"_"+t_dataset,step,loss_rate)
        result_degree_save_path = os.path.join(args.result_save_path, 'degree')
        folders = ['recon', 'gt', 'mask']
        for fold in folders:
            temp = os.path.join(args.result_save_path, fold)
            if not os.path.isdir(temp):
                os.makedirs(temp)
            temp = os.path.join(result_degree_save_path, fold)
            if not os.path.isdir(temp):
                os.makedirs(temp)

    if args.val:
        parts = args.data_path.split('/')
        dir_name, step, loss_rate = parts[-3], parts[-2], parts[-1]
        args.data_path = os.path.join(args.data_path, 'degree')
        args.performance_save_path = os.path.join(args.save_path, dir_name, step)
        args.loss_rate = loss_rate
        if not os.path.isdir(args.performance_save_path):
            os.makedirs(args.performance_save_path)
        
    return args