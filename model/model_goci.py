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
from tqdm import tqdm


class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = 0
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
            self.G = nn.DataParallel(self.G)

        self.optm_G = optim.Adam(self.G.parameters(), lr=1e-5)

        if train:
            self.writer = SummaryWriter(os.path.join("logs", os.path.basename(model_save_path)))
            self.lossNet = VGG16FeatureExtractor()

        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=1e-5)
                print('Model Initialized, iter:', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, starting from scratch')
            self.iter = 0

    def cuda(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        if self.lossNet is not None:
            self.lossNet.to(self.device)
        print(f"Model moved to {self.device}")

    def save_batch_images_grid(self, images, save_path, nrow=8, padding=2, normalize=True, is_mask=False):
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor but got {type(images)}")
        grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.endswith('.png'):
            save_path += '.png'
        save_image(grid, save_path)
        print(f"Image grid saved at {save_path}")

    def train(self, train_loader, save_path, store_capacity=10, finetune=False, iters=800000):
        self.G.train()
        s_time = time.time()
        while True:
            for items in train_loader:
                reiter_flag = False
                try:
                    if len(items) == 2:
                        gt_images, masks = self.__cuda__(*items)
                    elif len(items) == 3:
                        gt_images, masks, _ = self.__cuda__(*items)
                    else:
                        raise ValueError(f"Expected 2 or 3 items, got {len(items)}")
                except Exception as e:
                    print(f"[iter {self.iter}] Data loading error: {e}")
                    continue

                masks = (masks > 0).float()
                masked_images = gt_images * masks

                try:
                    _, fake_B, comp_B = self.forward(masked_images, masks, gt_images)
                except Exception as e:
                    print(f"[iter {self.iter}] Forward error: {e}")
                    continue

                if torch.isnan(fake_B).any() or torch.isnan(comp_B).any():
                    print(f"[iter {self.iter}] NaN in output — skipping")
                    continue

                loss_G = self.get_g_loss()
                if not torch.isfinite(loss_G):
                    print(f"[iter {self.iter}] Non-finite loss ({loss_G}) — skipping")
                    self.l1_loss_val = 0.0
                    continue

                try:
                    self.optm_G.zero_grad()
                    with torch.autograd.detect_anomaly():
                        loss_G.backward()
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                    self.optm_G.step()
                except Exception as e:
                    print(f"[iter {self.iter}] Backward error: {e}")
                    continue

                self.iter += 1

                if self.iter % 1000 == 0:
                    e_time = time.time()
                    print(f"Iter:{self.iter}, l1_loss:{self.l1_loss_val / 50:.4f}, time:{e_time - s_time:.2f}")
                    self.writer.add_scalar("Train/Loss", self.l1_loss_val, self.iter)
                    s_time = time.time()
                    self.l1_loss_val = 0.0

                if self.iter % 10000 == 0:
                    prefix = os.path.join(save_path, 'training', f"{os.path.basename(save_path)}_{self.iter}")
                    self.save_batch_images_grid(gt_images, f"{prefix}_gt")
                    self.save_batch_images_grid(comp_B, f"{prefix}_img")
                    self.save_batch_images_grid(fake_B, f"{prefix}_fake")
                    self.save_batch_images_grid(masked_images, f"{prefix}_masked")
                    self.save_batch_images_grid(masks, f"{prefix}_masks")
                    if is_available_to_store(store_capacity):
                        save_ckpt(f"{save_path}/g_{self.iter}.pth",
                                  [('generator', self.G)],
                                  [('optimizer_G', self.optm_G)],
                                  self.iter)

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
        perceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))

        for l in [tv_loss, style_loss, perceptual_loss, valid_loss, hole_loss]:
            if not torch.isfinite(l):
                print(f"[iter {self.iter}] Non-finite component detected — skipping")
                return torch.tensor(float('nan'), device=real_B.device)

        loss_G = (tv_loss * 0.1 + style_loss * 120 + perceptual_loss * 0.05 + valid_loss * 1 + hole_loss * 6)
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G

    def l1_loss(self, f1, f2, mask=1):
        diff = torch.abs(f1 - f2) * mask
        denom = torch.sum(mask)
        if denom == 0:
            return torch.tensor(0.0, device=f1.device)
        return torch.sum(diff) / denom

    def style_loss(self, A_feats, B_feats):
        loss_value = 0.0
        for A_feat, B_feat in zip(A_feats, B_feats):
            _, c, w, h = A_feat.size()
            A_flat = A_feat.view(A_feat.size(0), c, -1)
            B_flat = B_feat.view(B_feat.size(0), c, -1)
            A_style = torch.matmul(A_flat, A_flat.transpose(2, 1))
            B_style = torch.matmul(B_flat, B_flat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x, w_x = x.size(2), x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        return sum(torch.mean(torch.abs(a - b)) for a, b in zip(A_feats, B_feats))

    def Gray2VGGInput(self, x, dtype):
        x = torch.flatten(x, 1)
        x_min, _ = x.min(dim=1, keepdim=True)
        x_max, _ = x.max(dim=1, keepdim=True)
        eps = 1e-8
        x = ((x - x_min) / (x_max - x_min + eps)) * 255
        x = x.reshape(-1, 256, 256)

        x_group = []
        for i in range(x.shape[0]):
            x_img = Image.fromarray(np.uint8(x[i].cpu())).convert('RGB')
            x_tensor = self.totensor(x_img).to(x.device).unsqueeze(0)
            x_group.append(x_tensor)

        return torch.cat(x_group, dim=0)

    def __cuda__(self, *args):
        return [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in args]

    def extract_row_col(self, filename):
        match = re.search(r'_r(\d+)_c(\d+)', filename)
        if match:
            return match.group(1), match.group(2)
        raise ValueError(f"Invalid filename: {filename}")
