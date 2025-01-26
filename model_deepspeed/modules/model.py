# model_deepspeed/modules/model.py

import torch
import torch.nn as nn
import torch.optim as optim
from model_deepspeed.utils.io import load_ckpt, save_ckpt, is_available_to_store
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from model_deepspeed.modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re

class RFRNetModel(nn.Module):
    def __init__(self):
        super(RFRNetModel, self).__init__()
        self.G = RFRNet()
        self.lossNet = VGG16FeatureExtractor()
        self.iter = 0
        self.optm_G = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
        self.writer = None
        self.totensor = transforms.ToTensor()

    def initialize_model(self, path=None, train=True, model_save_path=None):
        # 모델과 손실 네트워크를 디바이스로 이동
        self.G = self.G.to(self.device)
        self.lossNet = self.lossNet.to(self.device)

        # TensorBoard 로그 디렉토리 설정
        if train:
            self.writer = SummaryWriter(os.path.join("logs", os.path.basename(model_save_path)))

        if train:
            self.lossNet = VGG16FeatureExtractor().to(self.device)

        if path:
            try:
                start_iter = load_ckpt(path, model_engine=None)  # DeepSpeed가 model_engine을 관리하므로 None 전달
                if train:
                    print('Model Initialized, iter:', start_iter)
                    self.iter = start_iter
            except Exception as e:
                print(f'No trained model, starting from scratch. Error: {e}')
                self.iter = 0

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        return masked_image, self.fake_B, self.comp_B

    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        perceptual_loss = self.perceptual_loss(real_B_feats, fake_B_feats) + self.perceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))

        loss_G = (tv_loss * 0.1
                  + style_loss * 120
                  + perceptual_loss * 0.05
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

    def perceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

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
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected images to be a torch.Tensor object, but got: {type(images)}")

        # Make a grid from the batch of images
        grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Ensure the file path includes a valid extension
        if not save_path.endswith('.png') and not save_path.endswith('.tiff'):
            save_path += '.png'

        # Save the grid as an image
        save_image(grid, save_path)
        print(f"Image grid saved at {save_path}")

    def update_parameters(self):
        loss_G = self.get_g_loss()
        loss_G.backward()
        if self.optm_G:
            self.optm_G.step()
            self.optm_G.zero_grad()

    def __cuda__(self, *args):
        result = []
        for item in args:
            if isinstance(item, torch.Tensor):
                result.append(item.to(self.device))
            else:
                result.append(item)  # If it's not a tensor (like a filename), just append it
        return result
