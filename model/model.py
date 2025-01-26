import torch
import torch.optim as optim
from utils.io import load_ckpt, save_ckpt, is_available_to_store
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import re

class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
        self.writer = None
        self.totensor = transforms.ToTensor()

    def initialize_model(self, path=None, train=True, model_save_path=None, gpu_ids=[0]):
        self.G = RFRNet()
        if torch.cuda.device_count() > 1:
            print(f"Using {len(gpu_ids)} GPUs in parallel.")
            self.G = nn.DataParallel(self.G)  # , device_ids=gpu_ids)

        self.optm_G = optim.Adam(self.G.parameters(), lr=2e-5)

        # TensorBoard log directory
        if train:
            self.writer = SummaryWriter(os.path.join("logs", os.path.basename(model_save_path)))

        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=2e-5)
                print('Model Initialized, iter:', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")

    def save_batch_images_grid(self, images, save_path, nrow=8, padding=2, normalize=True, is_mask=False):
        """
        Save a batch of images as a grid.
        - If is_mask=True: Save as grayscale PNG.
        - Else: Apply jet colormap to raw pixel values (0.01-10), normalize, and save as TIFF.

        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
            save_path (str): The path where the image grid should be saved (including extension).
            nrow (int): Number of images per row in the grid.
            padding (int): Amount of padding between images in the grid.
            normalize (bool): If True, normalize the tensor values to the 0-1 range for visualization.
            is_mask (bool): If True, save as grayscale PNG. Else, apply jet colormap and save as TIFF.
        """
        # print(f"Images shape: {images.shape}")  # Debugging: 출력
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected images to be a torch.Tensor object, but got: {type(images)}")

        # Make a grid from the batch of images
        grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Ensure the file path includes a valid extension (e.g., .png)
        if not save_path.endswith('.png'):
            save_path += '.png'

        # Save the grid as an image
        save_image(grid, save_path)
        print(f"Image grid saved at {save_path}")


    def train(self, train_loader, save_path, store_capacity=10, finetune=False, iters=800000):
        count = 0
        self.G.train()
        if finetune:
            for param in self.G.parameters():
                param.requires_grad = False
            for param in self.G.module.fc.parameters():  # assuming `fc` is the part you want to finetune
                param.requires_grad = True
            self.optm_G = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=5e-5)
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()

        while True:
            for items in train_loader:
                # Unpack the batch properly based on the number of items
                if len(items) == 2:
                    gt_images, masks = self.__cuda__(*items)
                elif len(items) == 3:  # If there is an additional item, like filenames
                    gt_images, masks, filenames = self.__cuda__(*items)
                else:
                    raise ValueError(f"Expected 2 or 3 items, but got {len(items)}")

                # Ensure the land-sea mask is applied before feeding the network
                masks = (masks > 0).float()  # Ensure the mask is a float tensor (0 for land, 1 for ocean)

                # Masked image generation
                masked_images = gt_images * masks

                # Forward pass
                masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_images)

                # Update parameters
                self.update_parameters()
                self.iter += 1

                if self.iter % 100 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" % (self.iter, self.l1_loss_val / 50, int_time))
                    self.writer.add_scalar("Train/Loss", self.l1_loss_val, self.iter)
                    s_time = time.time()
                    self.l1_loss_val = 0.0

                # Save image grid every 100 iterations
                if self.iter % 10000 == 0:
                    save_directory = os.path.join(save_path, 'training')
                    os.makedirs(save_directory, exist_ok=True)

                    file_prefix = f"{save_directory}/{os.path.basename(save_path)}_{self.iter}"
                    self.save_batch_images_grid(gt_images, f"{file_prefix}_gt")
                    self.save_batch_images_grid(comp_B, f"{file_prefix}_img")
                    self.save_batch_images_grid(fake_B, f"{file_prefix}_fake")
                    self.save_batch_images_grid(masked_image, f"{file_prefix}_masked")
                    self.save_batch_images_grid(masks, f"{file_prefix}_masks")

                # Save model checkpoint every 10000 iterations
                if self.iter % 10000 == 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if is_available_to_store(store_capacity):
                        save_ckpt(f'{save_path}/g_{self.iter}.pth', [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
                    else:
                        exit()

        # 이 부분은 while True 루프 내에 있으므로, 무한 루프이지만 안전하게 수정해야 합니다.
        # 현재 위치에서는 while True 루프가 종료되지 않아 아래 코드는 실행되지 않을 것입니다.
        # 따라서, 필요에 따라 루프를 종료하거나 조건을 추가해야 합니다.
        e_time = time.time()
        total_time = e_time - s_time
        print(f"Total time taken: {total_time:.2f}s")

    def extract_row_col(self, filename):
        """
        Extract row (r) and column (c) values from the filename.
        The filename is assumed to follow the pattern: *_r{row}_c{col}.tiff
        """
        match = re.search(r'_r(\d+)_c(\d+)', filename)
        if match:
            row = match.group(1)
            col = match.group(2)
            return row, col
        else:
            raise ValueError(f"Could not extract row and col from filename: {filename}")

    def test(self, test_loader, result_save_path):
        self.G.eval()  # Set the model to evaluation mode

        for para in self.G.parameters():
            para.requires_grad = False  # Disable gradient computation during testing

        count = 0
        s_time = time.time()

        # Create directories for saving results
        result_save_path_recon = os.path.join(result_save_path, 'recon')
        result_save_path_gt = os.path.join(result_save_path, 'gt')
        result_save_path_mask = os.path.join(result_save_path, 'mask')
        result_save_path_masked = os.path.join(result_save_path, 'masked')

        result_degree_save_path = os.path.join(result_save_path, 'degree')
        result_degree_save_path_gt = os.path.join(result_degree_save_path, 'gt')
        result_degree_save_path_mask = os.path.join(result_degree_save_path, 'mask')
        result_degree_save_path_recon = os.path.join(result_degree_save_path, 'recon')

        # Ensure the directories exist
        os.makedirs(result_save_path_recon, exist_ok=True)
        os.makedirs(result_save_path_gt, exist_ok=True)
        os.makedirs(result_save_path_mask, exist_ok=True)
        os.makedirs(result_save_path_masked, exist_ok=True)

        os.makedirs(result_degree_save_path, exist_ok=True)
        os.makedirs(result_degree_save_path_gt, exist_ok=True)
        os.makedirs(result_degree_save_path_mask, exist_ok=True)
        os.makedirs(result_degree_save_path_recon, exist_ok=True)

        # Iterate over the test_loader once
        for items in test_loader:
            # Unpack the batch properly based on the number of items
            if len(items) == 2:
                gt_images, masks = self.__cuda__(*items)
            elif len(items) == 3:  # If there is an additional item, like filenames
                gt_images, masks, filenames = self.__cuda__(*items)
            else:
                raise ValueError(f"Expected 2 or 3 items, but got {len(items)}")

            si_time = time.time()

            # Ensure the land-sea mask is applied before feeding the network
            masks = (masks > 0).float()  # Ensure the mask is a float tensor (0 for land, 1 for ocean)

            # Multiply the gt_images with land-sea mask to ensure only ocean parts are passed through
            masked_images = gt_images * masks

            # Forward pass: Use the masked image with the land-sea mask to exclude land during restoration
            masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_images)

            # Save images in grid format similar to training
            for k in range(fake_B.size(0)):
                count += 1

                # Get the filename and extract the row and column values
                filename = filenames[k]  # Assuming filenames are passed as part of the test_loader
                filename_no_ext = os.path.splitext(os.path.basename(filename))[0]

                # Define file prefixes for saving
                gt_file_prefix = f"{result_save_path_gt}/gt_{count}_{filename_no_ext}.png"
                mask_file_prefix = f"{result_save_path_mask}/mask_{count}_{filename_no_ext}.png"
                masked_file_prefix = f"{result_save_path_masked}/masked_{count}_{filename_no_ext}.png"
                recon_file_prefix = f"{result_save_path_recon}/recon_{count}_{filename_no_ext}.png"

                # Save images as grids
                self.save_batch_images_grid(gt_images[k:k+1], gt_file_prefix)         # Ground truth images
                self.save_batch_images_grid(masked_image[k:k+1], masked_file_prefix)  # Masked input images
                self.save_batch_images_grid(comp_B[k:k+1], recon_file_prefix)         # Reconstructed images
                self.save_batch_images_grid(masks[k:k+1], mask_file_prefix)           # Masks

                # Save the mask just like in the train() function
                mask_grid = masks[k:k+1]  # Extract the mask for the current image batch
                self.save_batch_images_grid(mask_grid, mask_file_prefix, nrow=1, padding=2, normalize=True)  # Save mask as a grid

                # Calculate degree values by averaging across channels and save as CSV files
                fake_degree = fake_B[k].mean(dim=0).cpu().numpy()  # Calculate mean across channels for degree
                gt_degree = gt_images[k, 1, :, :].cpu().numpy()    # Get the second channel (assuming it's needed for degree)
                mask_degree = masks[k, 1, :, :].cpu().numpy()      # Get the second channel for the mask

                # Save degree-related data (img, gt, mask) as CSV files
                file_path = f'{result_degree_save_path_recon}/img_{count}_{filename_no_ext}.csv'
                np.savetxt(file_path, fake_degree, delimiter=",")

                file_path = f'{result_degree_save_path_gt}/gt_{count}_{filename_no_ext}.csv'
                np.savetxt(file_path, gt_degree, delimiter=",")

                file_path = f'{result_degree_save_path_mask}/mask_{count}_{filename_no_ext}.csv'
                np.savetxt(file_path, mask_degree, delimiter=",")

            ei_time = time.time()
            i_time = ei_time - si_time
            print(f"Processed img#{count} in {i_time:.2f}s")

        # 이 부분은 while True 루프 내에 위치하고 있어 실행되지 않을 것입니다.
        # 필요에 따라 루프를 종료하거나 조건을 추가해야 합니다.
        e_time = time.time()
        total_time = e_time - s_time
        print(f"Total time taken: {total_time:.2f}s")

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        return masked_image, self.fake_B, self.comp_B

    def update_parameters(self):
        self.update_G()
        self.update_D()

    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()

    def update_D(self):
        return

    def Gray2VGGInput(self, x, dtype):
        #normalized [0-1]
        x = torch.flatten(x, 1)
        #x = x.unsqueeze(dim=1)
        x_min, _ = torch.min(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        #print(x_min.shape)
        x = ((x - x_min) / (x_max - x_min)) * 255
        #print(x.shape)
        x = x.reshape(6, 256, 256)
        #print(x.shape)

        x_group = []
        for i in range(6):
            x_img = x[i:i+1, :, :]
            x_img = torch.squeeze(x_img, 0)
            #print(x_img.shape)

            with torch.no_grad():
                x_img = Image.fromarray(np.uint8(x_img.cpu())).convert('RGB')
            x_tensor = self.totensor(x_img).cuda()
            x_tensor = torch.unsqueeze(x_tensor, 0)
            x_group.append(x_tensor)

        x_tensor = torch.cat(x_group, dim=0)

        #x_img = np.array(x_img)

        #x =  torch.cat([x]*3, dim = 1)
        #print(x.shape)
        return x_tensor

    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))

        loss_G = (  tv_loss * 0.1
                + style_loss * 120
                + preceptual_loss * 0.05
                + valid_loss * 1
                + hole_loss * 6)

        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G

    def l1_loss(self, f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def __cuda__(self, *args):
        result = []
        for item in args:
            if isinstance(item, torch.Tensor):
                result.append(item.to(self.device))
            else:
                result.append(item)  # If it's not a tensor (like a filename), just append it
        return result
