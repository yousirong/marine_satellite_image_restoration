import os
import torch
import json
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from model_withvalidation import RFRNetModel_withvalidation
from dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
import time

STATUS_FILE = 'validation_status.json'

def load_model_and_evaluate(pth_path, val_loader, train_loader, writer, step):
    model = RFRNetModel_withvalidation()
    model.initialize_model(path=pth_path, train=False)
    model.cuda()

    # Train Loss
    start_time = time.time()
    train_loss = model.validate(train_loader)  # Using validate function to calculate train loss
    end_time = time.time()
    print(f"Train loss for {pth_path} took {end_time - start_time:.2f} seconds")
    writer.add_scalar("Loss/Train", train_loss, step)

    # Validation Loss
    start_time = time.time()
    val_loss = model.validate(val_loader)
    end_time = time.time()
    print(f"Validation for {pth_path} took {end_time - start_time:.2f} seconds")
    writer.add_scalar("Loss/Validation", val_loss, step)

    

    return val_loss, train_loss

def save_status(directory, pth_file, step):
    status = {
        "directory": directory,
        "pth_file": pth_file,
        "step": step
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
        return status
    return None

def evaluate_all_models(directories, val_loader, train_loader, log_dir):
    writer = SummaryWriter(log_dir)
    status = load_status()
    start_directory = directories[0]
    start_file = None
    start_step = 0

    if status:
        start_directory = status['directory']
        start_file = status['pth_file']
        start_step = status['step']

    start_eval = False

    for directory in directories:
        if directory == start_directory:
            start_eval = True

        if not start_eval:
            continue

        pth_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        pth_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by the number in the filename
        pth_files = [f for f in pth_files if int(f.split('_')[1].split('.')[0]) % 1000 == 0]  # Select files with increments of 1000

        for pth_file in pth_files:
            step = int(pth_file.split('_')[1].split('.')[0])

            if directory == start_directory and start_file and pth_file <= start_file:
                continue

            pth_path = os.path.join(directory, pth_file)
            val_loss, train_loss = load_model_and_evaluate(pth_path, val_loader, train_loader, writer, step)
            print(f"Train Loss for {pth_file}: {train_loss:.4f}")
            print(f"Validation Loss for {pth_file}: {val_loss:.4f}")
            

            save_status(directory, pth_file, step)
    
    writer.close()

def create_data_loaders(data_root, mask_root, mask_mode, target_size, batch_size=64, num_workers=8, val_split=0.2):
    dataset = Dataset(data_root, mask_root, mask_mode, target_size, mask_reverse=False, training=True)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# Example usage
directories = [
    '/home/juneyonglee/MyData/backup_20240722/AY_ust/model/models/ust_chl_2',
    '/home/juneyonglee/MyData/backup_20240722/AY_ust/model/models/ust_chl_3',
    '/home/juneyonglee/MyData/backup_20240722/AY_ust/model/models/ust_chl_4'
]  # Update this to the paths of your .pth files directories
data_root = '/media/juneyonglee/My Book/data/Chl-a/train/perfect'  # Update this to the path of your validation data
mask_root = '/media/juneyonglee/My Book/data/mask/Train'  # Update this to the path of your validation masks
target_size = (256, 256)  # Update this to your target size
log_dir = 'model/models_validation/'  # Update this to your TensorBoard log directory

train_loader, val_loader = create_data_loaders(data_root, mask_root, mask_mode=0, target_size=target_size, batch_size=128, num_workers=8, val_split=0.2)

evaluate_all_models(directories, val_loader, train_loader, log_dir)
