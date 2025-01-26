import torch
import deepspeed
from model_deepspeed.utils.config import get_config
from model_deepspeed.utils.dataset import Dataset
from model_deepspeed.modules.model import RFRNetModel
from model_deepspeed.utils.io import save_ckpt, load_ckpt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

def train():
    # Load configuration
    cfg = get_config()

    # Initialize distributed training
    deepspeed.init_distributed()

    # Get the local rank for the current process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # Initialize the dataset and data loader with DistributedSampler
    train_dataset = Dataset(
        image_path=cfg['data_root'],
        mask_path=cfg['mask_path'],
        land_sea_mask_path=cfg['land_sea_mask_path'],
        mask_mode=cfg['mask_mode'],
        target_size=cfg['target_size'],
        augment=True,
        training=True,
        mask_reverse=cfg['mask_reverse']
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,  # Shuffle is handled by DistributedSampler
        sampler=train_sampler,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # Initialize the model
    model = RFRNetModel()
    model.initialize_model(
        path=cfg.get('model_path', None),
        train=True,
        model_save_path=cfg['model_save_path']
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=cfg['deepspeed_config']
    )

    # Load checkpoint if provided
    if cfg.get('model_path'):
        tag = os.path.splitext(os.path.basename(cfg['model_path']))[0]
        n_iter = load_ckpt(model_engine, cfg['model_save_path'], tag=tag)
        model.iter = n_iter
    else:
        model.iter = 0

    # Set model to training mode
    model_engine.train()

    # ===============================
    # 여기서 모델의 dtype을 추론
    # ===============================
    param_dtype = next(model_engine.parameters()).dtype
    # 또는
    # param_dtype = next(model_engine.module.parameters()).dtype

    # TensorBoard SummaryWriter 초기화
    log_dir = cfg.get('tensorboard_log_dir', './runs')
    writer = None
    if local_rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    # Training loop over epochs
    global_step = 0
    for epoch in range(cfg['num_epochs']):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")):
            imgs, masks, *rest = batch

            # 모델 파라미터와 동일한 dtype으로 변환
            imgs = imgs.to(device=device, dtype=param_dtype)
            masks = masks.to(device=device, dtype=param_dtype)

            # Forward pass
            masked_images = imgs * masks
            masked_image, fake_B, comp_B = model_engine(masked_images, masks, imgs)

            # Compute loss
            loss = model_engine.get_g_loss()

            # Backward pass and optimization
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()

            # TensorBoard 로깅
            if local_rank == 0 and writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/Epoch', epoch + 1, global_step)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Train/LearningRate', current_lr, global_step)

            global_step += 1

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)

        # Only the master process prints and saves checkpoints
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{cfg['num_epochs']}], Loss: {avg_loss:.4f}")

            if writer is not None:
                writer.add_scalar('Train/EpochLoss', avg_loss, epoch + 1)

            # Save checkpoint every few epochs (e.g., every 10 epochs)
            if (epoch + 1) % 10 == 0:
                ckpt_tag = f"epoch_{epoch+1}"
                save_ckpt(model_engine, cfg['model_save_path'], tag=ckpt_tag)
                print(f"Checkpoint saved at {cfg['model_save_path']} with tag {ckpt_tag}")

            # Save images every epoch
            save_images_at_epoch(model, epoch + 1, cfg['model_save_path'])

    if local_rank == 0:
        print("Training completed.")
        if writer is not None:
            writer.close()

def save_images_at_epoch(model, epoch, save_path):
    """
    Function to save masked, fake, and ground truth images at the end of each epoch.
    """
    if model.real_A is not None and model.fake_B is not None and model.real_B is not None and model.mask is not None:
        save_directory = os.path.join(save_path, f'train_epoch_{epoch}')
        os.makedirs(save_directory, exist_ok=True)

        model.save_batch_images_grid(model.real_A, os.path.join(save_directory, f'gt_epoch_{epoch}.png'))
        model.save_batch_images_grid(model.fake_B, os.path.join(save_directory, f'fake_epoch_{epoch}.png'))
        model.save_batch_images_grid(model.mask, os.path.join(save_directory, f'mask_epoch_{epoch}.png'))
        model.save_batch_images_grid(model.comp_B, os.path.join(save_directory, f'comp_epoch_{epoch}.png'))
        print(f"Saved images for epoch {epoch} at {save_directory}")

if __name__ == "__main__":
    train()
