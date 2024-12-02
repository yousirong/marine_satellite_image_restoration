import argparse
import os
from model import RFRNetModel
from model_withvalidation import RFRNetModel_withvalidation
from dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.io import load_yaml, build_save_folder
import torch
# GOCI 1day new validate function
from val_chl import validate

# ust21 8day new validate function
# from val_chl_new import validate


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default="", type=str, help="config file path")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()

    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    args = build_save_folder(args)

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))  # args.gpu_ids를 이용해 CUDA_VISIBLE_DEVICES 설정

    model = RFRNetModel()

    if args.test:
        model.initialize_model(args.model_path, False, None, args.gpu_ids)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root,args.land_sea_mask_path, args.mask_mode, args.target_size, mask_reverse=False, training=False),
                                batch_size=args.batch_size,
                                num_workers=args.n_threads)  # num_workers 설정
        model.test(dataloader, args.result_save_path)

    elif args.val:
        validate(args.loss_rate, args.data_path, args.performance_save_path, args.land_sea_mask_path)

    else:
        model.initialize_model(args.model_path, True, args.model_save_path, args.gpu_ids)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.land_sea_mask_path, args.mask_mode, args.target_size, mask_reverse=False),
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_threads)  # num_workers 설정
        model.train(dataloader, args.model_save_path, args.save_capacity, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()
