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

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        # self.fake_B = fake_B
        self.fake_B = torch.nan_to_num(fake_B, nan=0.0, posinf=1e3, neginf=-1e3)

        # self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        comp = self.fake_B * (1 - mask) + self.real_B * mask
        self.comp_B = torch.nan_to_num(comp, nan=0.0, posinf=1e3, neginf=-1e3)
        return masked_image, self.fake_B, self.comp_B



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
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected images to be a torch.Tensor object, but got: {type(images)}")

        # recon 이미지인 경우 RGB -> Grayscale 평균
        if images.size(1) == 3 and "recon" in save_path:
            # 평균을 내서 (B, 1, H, W) 로 만듦
            images = images.mean(dim=1, keepdim=True)

        # 그리드 만들기 (normalize 필요 시)
        if normalize:
            grid = make_grid(images, nrow=nrow, padding=padding, normalize=True)
        else:
            # 직접 정규화
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
            grid = make_grid(images, nrow=nrow, padding=padding, normalize=False)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.endswith('.png'):
            save_path += '.png'

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
                # ust21 일경우
                # masked_images = gt_images * masks
                #goci 일경우
                masked_images = gt_images * masks
                # Forward pass
                masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_images)
                if torch.isnan(fake_B).any():
                    print("Warning: NaN in network output (fake_B)!")

                if torch.isnan(comp_B).any():
                    print("Warning: NaN in composited output (comp_B)!")
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

                # Save image grid every 10000 iterations
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
        self.G.eval()  # 모델 평가 모드로 전환
        for para in self.G.parameters():
            para.requires_grad = False  # 테스트 시 gradient 계산 비활성화

        # 전달받은 result_save_path를 절대 경로로 변환하여 사용
        result_save_path = os.path.abspath(result_save_path)
        print("Saving test results to:", result_save_path)

        count = 0
        s_time = time.time()

        # 전체 테스트 이미지 개수를 계산 (test_loader.dataset이 존재하는 경우)
        total_test_images = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0
        if total_test_images == 0:
            print("전체 테스트 이미지 개수를 확인할 수 없습니다.")

        # 결과 저장 디렉터리 생성 (원하는 경로 내의 하위 폴더들)
        result_save_path_recon = os.path.join(result_save_path, 'recon')
        result_save_path_gt = os.path.join(result_save_path, 'gt')
        result_save_path_mask = os.path.join(result_save_path, 'mask')
        result_save_path_masked = os.path.join(result_save_path, 'masked')

        result_degree_save_path = os.path.join(result_save_path, 'degree')
        result_degree_save_path_gt = os.path.join(result_degree_save_path, 'gt')
        result_degree_save_path_mask = os.path.join(result_degree_save_path, 'mask')
        result_degree_save_path_recon = os.path.join(result_degree_save_path, 'recon')

        for d in [result_save_path_recon, result_save_path_gt, result_save_path_mask,
                result_save_path_masked, result_degree_save_path, result_degree_save_path_gt,
                result_degree_save_path_mask, result_degree_save_path_recon]:
            os.makedirs(d, exist_ok=True)

        # tqdm 진행바 생성: 전체 테스트 이미지 개수를 총합으로 설정
        pbar = tqdm(total=total_test_images, desc="Processing test images")

        # 테스트 데이터셋의 배치를 순회
        for items in test_loader:
            # items에 filename 정보가 없는 경우, 임의로 생성
            if len(items) == 2:
                gt_images, masks = self.__cuda__(*items)
                batch_size = gt_images.size(0)
                filenames = [f"test_{i}" for i in range(batch_size)]
            elif len(items) == 3:
                gt_images, masks, filenames = self.__cuda__(*items)
                batch_size = gt_images.size(0)
            else:
                raise ValueError(f"Expected 2 or 3 items, but got {len(items)}")

            si_time = time.time()

            # 마스크를 0/1 float 텐서로 변환 (0: 육지, 1: 해양)
            masks = (masks > 0).float()
            # gt_images와 마스크를 곱하여 해양 영역만 추출
            masked_images = gt_images * masks

            # forward pass (train과 동일한 방식으로 처리)
            masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_images)

            # 배치 내 각 샘플별 결과 저장
            for k in range(batch_size):
                count += 1
                # filename이 이미 있다면 사용, 없으면 임의의 이름을 사용
                filename = filenames[k]
                filename_no_ext = os.path.splitext(os.path.basename(filename))[0]

                # 저장할 파일 경로 정의 (PNG 형식으로 저장)
                gt_file_prefix = os.path.join(result_save_path_gt, f"gt_{count}_{filename_no_ext}.png")
                mask_file_prefix = os.path.join(result_save_path_mask, f"mask_{count}_{filename_no_ext}.png")
                masked_file_prefix = os.path.join(result_save_path_masked, f"masked_{count}_{filename_no_ext}.png")
                recon_file_prefix = os.path.join(result_save_path_recon, f"recon_{count}_{filename_no_ext}.png")

                # 결과 이미지를 그리드 형식으로 저장
                self.save_batch_images_grid(gt_images[k:k+1], gt_file_prefix)
                self.save_batch_images_grid(masked_image[k:k+1], masked_file_prefix)
                self.save_batch_images_grid(comp_B[k:k+1], recon_file_prefix)
                self.save_batch_images_grid(masks[k:k+1], mask_file_prefix)

                # 채널별 평균(예: degree 계산) 데이터를 CSV로 저장 (필요한 경우)
                fake_degree = fake_B[k].mean(dim=0).cpu().numpy()  # 채널 평균
                gt_degree = gt_images[k, 1, :, :].cpu().numpy()      # 두 번째 채널 (예시)

                # 기존 mask_degree (원래는 0: 결측, 1: 유효) 추출
                mask_degree = masks[k, 1, :, :].cpu().numpy()
                # gt_images의 두 번째 채널을 기준으로 육지와 해양을 구분 (해양: 값 > 0)
                gt_channel = gt_images[k, 1, :, :].cpu().numpy()
                ocean_region = (gt_channel > 0)
                # 육지 영역(해양이 아닌 부분)은 결측치 계산에서 제외하기 위해 유효한 값(예: 1)으로 설정
                mask_degree[~ocean_region] = 1

                np.savetxt(os.path.join(result_degree_save_path_recon, f"img_{count}_{filename_no_ext}.csv"),
                        fake_degree, delimiter=",")
                np.savetxt(os.path.join(result_degree_save_path_gt, f"gt_{count}_{filename_no_ext}.csv"),
                        gt_degree, delimiter=",")
                np.savetxt(os.path.join(result_degree_save_path_mask, f"mask_{count}_{filename_no_ext}.csv"),
                        mask_degree, delimiter=",")

            ei_time = time.time()
            elapsed_time = ei_time - s_time  # 배치 처리에 걸린 시간
            # 배치 처리 후 진행바 업데이트
            pbar.update(batch_size)
            # 진행바에 경과 시간 및 현재 진행률을 postfix로 표시
            percent = (count / total_test_images) * 100 if total_test_images > 0 else 0
            pbar.set_postfix({"elapsed": f"{elapsed_time:.2f}s", "percent": f"{percent:.2f}%"})

        pbar.close()
        print("Test completed: 100%")

    def update_parameters(self):
        self.update_G()
        self.update_D()

    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()

        # Gradient clipping 추가 (클립 임계값: 1.0)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

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
        # TV 는 복원(홀) 영역에서만
        tv_loss         = self.TV_loss(comp_B * self.mask)

        # hole_loss: mask==1 (검은 영역) 에 대한 L1
        hole_loss       = self.l1_loss(real_B, fake_B, self.mask)
        # valid_loss: mask==0 (유효 영역) 에 대한 L1
        valid_loss      = self.l1_loss(real_B, fake_B, (1 - self.mask))
        style_loss      = self.style_loss(real_B_feats, fake_B_feats) \
                        + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) \
                        + self.preceptual_loss(real_B_feats, comp_B_feats)

        loss_G = (  tv_loss * 0.1
                + style_loss * 120
                + preceptual_loss * 0.05
                + valid_loss *   1
                + hole_loss  *   6)

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