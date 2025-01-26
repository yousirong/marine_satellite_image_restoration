# utils/config.py
import argparse
import yaml
import os
from ds_config import ds_config  # DeepSpeed 설정 임포트

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config():
    parser = argparse.ArgumentParser(description="Training and Inference Configuration")

    # Config file
    parser.add_argument("--config", type=str, required=False, help="Path to the YAML configuration file.")

    # Override specific parameters if needed
    parser.add_argument("--model_save_path", type=str, help="Path to save model checkpoints.")
    parser.add_argument("--data_root", type=str, help="Path to the dataset root.")
    parser.add_argument("--mask_path", type=str, help="Path to the mask files.")
    parser.add_argument("--land_sea_mask_path", type=str, help="Path to the land-sea mask .npy file.")
    parser.add_argument("--mask_mode", type=str, help="Mask mode.")
    parser.add_argument("--target_size", type=int, help="Target size for images.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0, 1], help="GPU IDs to use.")
    parser.add_argument("--mask_reverse", type=str2bool, nargs='?', const=True, default=False, help="Whether to reverse the mask.")
    parser.add_argument("--save_capacity", type=int, default=5, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader threads.")

    # Add local_rank for DeepSpeed
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training.")

    # Flags
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune the model.")
    parser.add_argument('--test', action='store_true', help="Whether to run in test mode.")
    parser.add_argument('--val', action='store_true', help="Whether to run in validation mode.")

    args = parser.parse_args()

    # Initialize a configuration dictionary
    cfg = {}

    # If a YAML config file is provided, load it
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    # Override config with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key != "config":
            cfg[key] = value

    # Merge DeepSpeed config
    cfg['deepspeed_config'] = ds_config

    # Ensure paths are absolute
    if 'model_save_path' in cfg:
        cfg['model_save_path'] = os.path.abspath(cfg['model_save_path'])
    if 'data_root' in cfg:
        cfg['data_root'] = os.path.abspath(cfg['data_root'])
    if 'mask_root' in cfg:
        cfg['mask_root'] = os.path.abspath(cfg['mask_root'])
    if 'land_sea_mask_path' in cfg:
        cfg['land_sea_mask_path'] = os.path.abspath(cfg['land_sea_mask_path'])

    return cfg
