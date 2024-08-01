import os
import torch
from torch.utils.data import DataLoader
from model_withvalidation import RFRNetModel_withvalidation
from dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

def load_model_and_evaluate(pth_path, val_loader, writer, step):
    model = RFRNetModel_withvalidation()
    model.initialize_model(path=pth_path, train=False)
    model.cuda()
    val_loss = model.validate(val_loader)
    writer.add_scalar("Validation/Loss", val_loss, step)
    return val_loss

def evaluate_all_models(directories, val_loader, log_dir):
    writer = SummaryWriter(log_dir)
    for directory in directories:
        pth_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        pth_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by the number in the filename
        pth_files = [f for f in pth_files if int(f.split('_')[1].split('.')[0]) % 100 == 0]  # Select files with increments of 100

        for pth_file in pth_files:
            step = int(pth_file.split('_')[1].split('.')[0])
            pth_path = os.path.join(directory, pth_file)
            val_loss = load_model_and_evaluate(pth_path, val_loader, writer, step)
            print(f"Validation Loss for {pth_file}: {val_loss:.4f}")
    
    writer.close()

def create_val_loader(data_root, mask_root, mask_mode, target_size, batch_size=32, num_workers=4):
    val_dataset = Dataset(data_root, mask_root, mask_mode, target_size, mask_reverse=False, training=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_loader

# Example usage
directories = [
    '/home/juneyonglee/MyData/backup_20240722/AY_ust/model/models/ust_chl_2',
    '/home/juneyonglee/MyData/backup_20240722/AY_ust/model/models/ust_chl_3',
    '/home/juneyonglee/MyData/backup_20240722/AY_ust/model/models/ust_chl_4'
]  # Update this to the paths of your .pth files directories
data_root = '/media/juneyonglee/My Book/data/Chl-a/train/perfect'  # Update this to the path of your validation data
mask_root = '/media/juneyonglee/My Book/data/mask/Train'  # Update this to the path of your validation masks
mask_mode = 2  # Update this to the specific mask mode
target_size = (256, 256)  # Update this to your target size
log_dir = 'model_save_path: model/models_validation/'  # Update this to your TensorBoard log directory

val_loader = create_val_loader(data_root, mask_root, mask_mode, target_size)
evaluate_all_models(directories, val_loader, log_dir)
