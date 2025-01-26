# scripts/inference.py
import torch
import deepspeed
from model_deepspeed.utils.config import get_config
from utils.data_loader import Dataset
from model_deepspeed.modules.model import RFRNetModel
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pandas as pd

def inference():
    # Load configuration
    cfg = get_config()

    # Initialize the dataset and data loader
    test_dataset = Dataset(
        image_path=cfg['data_root'],
        mask_path=cfg['mask_path'],
        land_sea_mask_path=cfg['land_sea_mask_path'],
        mask_mode=cfg['mask_mode'],
        target_size=cfg['target_size'],
        augment=False,
        training=False,
        mask_reverse=cfg['mask_reverse']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['n_threads'],
        pin_memory=True
    )

    # Initialize the model
    model = RFRNetModel()
    model.initialize_model(
        path=cfg['model_path'],
        train=False,
        model_save_path=cfg['model_save_path'],
        gpu_ids=cfg['gpu_ids']
    )

    # Move model to device and initialize DeepSpeed
    model.cuda()

    # Initialize DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        model=model.G,
        optimizer=model.optm_G,
        config=cfg['deepspeed_config']
    )

    # Load the latest checkpoint
    checkpoint_dir = cfg['model_save_path']
    checkpoint = deepspeed.checkpointing.load_checkpoint(checkpoint_dir, tag="epoch_last")

    # Set model to evaluation mode
    model_engine.eval()

    predictions = []
    filenames = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            imgs, masks, filenames_batch = batch
            imgs = imgs.to(model_engine.local_rank)
            masks = masks.to(model_engine.local_rank).float()

            # Forward pass
            fake_B, comp_B = model_engine(imgs, masks, imgs)  # Adjust based on your forward method

            # Collect predictions
            preds = fake_B.cpu().numpy()
            predictions.extend(preds)
            filenames.extend(filenames_batch)

    # Save predictions to CSV
    df = pd.DataFrame({
        'filename': filenames,
        'prediction': predictions
    })
    save_csv_path = os.path.join(cfg['model_save_path'], 'inference_results.csv')
    df.to_csv(save_csv_path, index=False)
    print(f"Inference completed and results saved at {save_csv_path}")

if __name__ == "__main__":
    inference()
