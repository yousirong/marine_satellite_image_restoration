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

        # 실제 데이터 분포에 맞춘 전처리 파라미터
        self.missing_value = -999   # 결측값 표시
        self.valid_min = -1000      # 유효한 데이터 최소값
        self.valid_max = 100        # 유효한 데이터 최대값
        self.normalize_min = -900   # 정규화 최소값
        self.normalize_max = 50     # 정규화 최대값

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

    def preprocess_rrs_data(self, data, mode='adaptive_fill'):
        """
        실제 RRS 데이터 분포에 맞춘 전처리
        """
        processed_data = data.clone()

        # 1. 결측값(-999) 마스크 생성
        missing_mask = (processed_data == self.missing_value)

        # 2. 매우 이상한 값들 제거
        extreme_mask = (processed_data < -1000) | (processed_data > 1000)

        # 3. 전체 무효값 마스크
        invalid_mask = missing_mask | extreme_mask

        if mode == 'adaptive_fill':
            # 배치별, 채널별 적응적 채우기
            for b in range(data.size(0)):
                for c in range(data.size(1)):
                    channel_data = processed_data[b, c]
                    channel_invalid = invalid_mask[b, c]

                    # 유효한 값들만 추출
                    valid_values = channel_data[~channel_invalid]

                    if len(valid_values) > 0:
                        # 유효한 값들의 분포 분석
                        valid_sorted = torch.sort(valid_values)[0]
                        n_valid = len(valid_sorted)

                        if n_valid >= 10:
                            # 충분한 데이터가 있으면 10~90 percentile 사용
                            p10_idx = max(0, n_valid // 10)
                            p90_idx = min(n_valid - 1, (n_valid * 9) // 10)
                            fill_min = valid_sorted[p10_idx]
                            fill_max = valid_sorted[p90_idx]

                            # 범위 내에서 랜덤하게 채우기
                            fill_range = fill_max - fill_min
                            if fill_range > 0:
                                random_fill = fill_min + torch.rand_like(channel_data[channel_invalid]) * fill_range
                            else:
                                random_fill = torch.full_like(channel_data[channel_invalid], fill_min)
                            processed_data[b, c][channel_invalid] = random_fill
                        else:
                            # 데이터가 적으면 median으로 채우기
                            median_value = torch.median(valid_values)
                            processed_data[b, c][channel_invalid] = median_value
                    else:
                        # 모든 값이 무효한 경우 0으로 설정
                        processed_data[b, c][channel_invalid] = 0.0

        elif mode == 'zero_fill':
            processed_data[invalid_mask] = 0.0

        elif mode == 'median_fill':
            valid_values = processed_data[~invalid_mask]
            if len(valid_values) > 0:
                global_median = torch.median(valid_values)
                processed_data[invalid_mask] = global_median
            else:
                processed_data[invalid_mask] = 0.0

        return processed_data, ~invalid_mask  # 유효한 픽셀 마스크도 반환

    def smart_normalize(self, data, valid_mask=None):
        """
        실제 데이터 분포를 고려한 정규화
        """
        if valid_mask is not None:
            valid_data = data[valid_mask]
            if len(valid_data) > 0:
                # 실제 데이터 범위 확인
                data_min = torch.min(valid_data)
                data_max = torch.max(valid_data)

                # 데이터 분포에 맞는 정규화 범위 설정
                if data_min < -500:  # 큰 음수 값들이 있는 경우
                    norm_min = max(data_min.item(), -900)  # -900 이하는 클리핑
                    norm_max = min(data_max.item(), 100)   # 100 이상은 클리핑
                else:
                    norm_min = data_min.item()
                    norm_max = data_max.item()

                # 범위가 너무 작으면 기본값 사용
                if (norm_max - norm_min) < 1e-3:
                    norm_min, norm_max = -1.0, 1.0

                # Min-Max 정규화 (0-1 범위로)
                normalized = torch.clamp((data - norm_min) / (norm_max - norm_min), 0, 1)
                return normalized, norm_min, norm_max

        # fallback: 기본 정규화
        return torch.clamp((data + 900) / 950, 0, 1), -900.0, 50.0

    def save_batch_images_grid(self, images, save_path, nrow=8, padding=2, normalize=True, is_mask=False):
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor but got {type(images)}")

        # NaN/Inf 체크
        if torch.isnan(images).any():
            print("  WARNING: Input contains NaN values!")
        if torch.isinf(images).any():
            print("  WARNING: Input contains Inf values!")
        if torch.all(images == 0):
            print("  WARNING: All values are zero!")

        try:
            grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not save_path.endswith('.png'):
                save_path += '.png'
            save_image(grid, save_path)
        except Exception as e:
            print(f"  Error during save: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def train(self, train_loader, save_path, store_capacity=10, finetune=False, iters=800000):
        self.G.train()
        s_time = time.time()

        # 손실 통계를 위한 변수들
        total_loss_count = 0
        valid_loss_sum = 0
        skipped_batches = 0

        while True:
            for items in train_loader:
                try:
                    if len(items) == 2:
                        gt_images, masks = self.__cuda__(*items)
                    elif len(items) == 3:
                        gt_images, masks, _ = self.__cuda__(*items)
                    else:
                        raise ValueError(f"Expected 2 or 3 items, got {len(items)}")
                except Exception as e:
                    print(f"[iter {self.iter}] Data loading error: {e}")
                    skipped_batches += 1
                    continue

                # ===== 실제 데이터 분포에 맞춘 전처리 =====
                # 1. GT 이미지 전처리 (적응적 채우기 방식 사용)
                gt_processed, gt_valid_mask = self.preprocess_rrs_data(
                    gt_images, mode='adaptive_fill'
                )

                # 2. 마스크 이진화
                masks = (masks > 0).float()

                # 3. 유효한 픽셀 비율 확인
                valid_pixel_ratio = gt_valid_mask.float().mean()
                if valid_pixel_ratio < 0.05:  # 5% 이상만 있으면 OK
                    print(f"[iter {self.iter}] Insufficient valid pixels ({valid_pixel_ratio:.3f}) - skipping")
                    skipped_batches += 1
                    continue

                # 4. 실제 데이터 분포에 맞춘 정규화
                gt_normalized, data_min, data_max = self.smart_normalize(gt_processed, gt_valid_mask)

                # 첫 번째 배치에서 데이터 통계 출력
                if self.iter % 1000 == 0:
                    print(f"[iter {self.iter}] Data stats:")
                    print(f"  Original range: [{gt_images.min():.2f}, {gt_images.max():.2f}]")
                    print(f"  Processed range: [{gt_processed.min():.2f}, {gt_processed.max():.2f}]")
                    print(f"  Normalized range: [{gt_normalized.min():.2f}, {gt_normalized.max():.2f}]")
                    print(f"  Valid pixel ratio: {valid_pixel_ratio:.3f}")
                    print(f"  Normalization params: min={data_min:.2f}, max={data_max:.2f}")

                # 5. 마스킹된 이미지 생성
                masked_images = gt_normalized * masks

                # ===== 데이터 품질 체크 =====
                if (torch.isnan(gt_normalized).any() or torch.isnan(masked_images).any() or
                    torch.isinf(gt_normalized).any() or torch.isinf(masked_images).any()):
                    print(f"[iter {self.iter}] NaN/Inf after preprocessing - skipping")
                    skipped_batches += 1
                    continue

                try:
                    # Forward pass
                    _, fake_B, comp_B = self.forward(masked_images, masks, gt_normalized)

                    # 출력 품질 체크
                    if torch.isnan(fake_B).any() or torch.isnan(comp_B).any():
                        print(f"[iter {self.iter}] NaN in model output - skipping")
                        skipped_batches += 1
                        continue
                except Exception as e:
                    print(f"[iter {self.iter}] Forward error: {e}")
                    skipped_batches += 1
                    continue

                # Loss 계산 (유효한 픽셀에 대해서만)
                loss_G = self.get_robust_loss(gt_valid_mask)

                if not torch.isfinite(loss_G):
                    print(f"[iter {self.iter}] Non-finite loss ({loss_G}) - skipping")
                    skipped_batches += 1
                    continue

                try:
                    # Backward pass
                    self.optm_G.zero_grad()
                    loss_G.backward()

                    # Gradient clipping (더 강한 클리핑)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=0.5)

                    self.optm_G.step()

                    # 통계 업데이트
                    total_loss_count += 1
                    valid_loss_sum += loss_G.item()
                except Exception as e:
                    print(f"[iter {self.iter}] Backward error: {e}")
                    skipped_batches += 1
                    continue

                self.iter += 1

                # 로그 출력
                if self.iter % 1000 == 0:
                    e_time = time.time()
                    avg_loss = valid_loss_sum / max(total_loss_count, 1)
                    skip_rate = skipped_batches / max(self.iter, 1) * 100

                    print(f"Iter:{self.iter}, avg_loss:{avg_loss:.4f}, "
                          f"skip_rate:{skip_rate:.1f}%, time:{e_time - s_time:.2f}s")

                    self.writer.add_scalar("Train/Loss", avg_loss, self.iter)
                    self.writer.add_scalar("Train/SkipRate", skip_rate, self.iter)
                    self.writer.add_scalar("Train/ValidPixelRatio", valid_pixel_ratio, self.iter)

                    s_time = time.time()
                    # 통계 리셋
                    total_loss_count = 0
                    valid_loss_sum = 0
                    skipped_batches = 0

                # 이미지 저장
                if self.iter % 5000 == 0:  # 더 적은 빈도로 저장
                    prefix = os.path.join(save_path, 'training', f"{os.path.basename(save_path)}_{self.iter}")
                    try:
                        self.save_batch_images_grid(gt_normalized, f"{prefix}_gt")
                        self.save_batch_images_grid(comp_B, f"{prefix}_img")
                        self.save_batch_images_grid(masked_images, f"{prefix}_masked")
                        self.save_batch_images_grid(masks, f"{prefix}_masks")
                    except Exception as e:
                        print(f"Error saving images: {e}")

                # 모델 저장
                if self.iter % 10000 == 0:
                    if is_available_to_store(store_capacity):
                        save_ckpt(f"{save_path}/g_{self.iter}.pth",
                                  [('generator', self.G)],
                                  [('optimizer_G', self.optm_G)],
                                  self.iter)

                if self.iter >= iters:
                    return

    def get_robust_loss(self, valid_mask=None):
        """
        유효한 픽셀에 대해서만 loss 계산
        """
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        mask = self.mask

        # 기본 loss 계산
        try:
            real_B_feats = self.lossNet(real_B)
            fake_B_feats = self.lossNet(fake_B)
            comp_B_feats = self.lossNet(comp_B)

            # TV loss (유효한 영역에서만)
            if valid_mask is not None:
                tv_loss = self.TV_loss_masked(comp_B * (1 - mask), valid_mask)
            else:
                tv_loss = self.TV_loss(comp_B * (1 - mask))

            # Style and perceptual loss
            style_loss = (self.style_loss(real_B_feats, fake_B_feats) +
                         self.style_loss(real_B_feats, comp_B_feats))
            perceptual_loss = (self.preceptual_loss(real_B_feats, fake_B_feats) +
                              self.preceptual_loss(real_B_feats, comp_B_feats))

            # L1 loss (유효한 픽셀에서만)
            if valid_mask is not None:
                valid_loss = self.l1_loss_masked(real_B, fake_B, mask, valid_mask)
                hole_loss = self.l1_loss_masked(real_B, fake_B, (1 - mask), valid_mask)
            else:
                valid_loss = self.l1_loss(real_B, fake_B, mask)
                hole_loss = self.l1_loss(real_B, fake_B, (1 - mask))

            # NaN/Inf 체크
            losses = [tv_loss, style_loss, perceptual_loss, valid_loss, hole_loss]
            for i, l in enumerate(losses):
                if not torch.isfinite(l):
                    print(f"Non-finite loss component {i}: {l}")
                    return torch.tensor(float('nan'), device=real_B.device)

            # 가중치 조정 (더 보수적인 가중치)
            loss_G = (tv_loss * 0.05 + style_loss * 60 + perceptual_loss * 0.02 +
                     valid_loss * 1 + hole_loss * 3)

            self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
            return loss_G

        except Exception as e:
            print(f"Error in loss calculation: {e}")
            return torch.tensor(float('nan'), device=real_B.device)

    def l1_loss_masked(self, f1, f2, mask, valid_mask):
        """
        유효한 픽셀에 대해서만 L1 loss 계산
        """
        combined_mask = mask * valid_mask  # 두 마스크의 교집합
        diff = torch.abs(f1 - f2) * combined_mask
        denom = torch.sum(combined_mask)

        if denom == 0:
            return torch.tensor(0.0, device=f1.device)
        return torch.sum(diff) / denom

    def TV_loss_masked(self, x, valid_mask):
        """
        유효한 픽셀에 대해서만 TV loss 계산
        """
        h_x, w_x = x.size(2), x.size(3)

        # Horizontal TV
        h_diff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        h_valid = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
        h_tv = torch.sum(h_diff * h_valid) / torch.sum(h_valid).clamp(min=1)

        # Vertical TV
        w_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        w_valid = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
        w_tv = torch.sum(w_diff * w_valid) / torch.sum(w_valid).clamp(min=1)

        return h_tv + w_tv

    def test(self, test_loader, result_save_path):
        """
        Test the RFRNet model with RRS-aware preprocessing
        """
        print("=== Model Health Check ===")
        model_healthy = self.check_model_health()

        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False

        result_save_path = os.path.abspath(result_save_path)
        print("Saving test results to:", result_save_path)

        count = 0
        skipped_count = 0
        generator_success_count = 0
        fallback_count = 0
        s_time = time.time()

        total_test_images = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0
        if total_test_images == 0:
            print("전체 테스트 이미지 개수를 확인할 수 없습니다.")

        # 결과 저장 디렉터리 생성
        result_save_path_recon = os.path.join(result_save_path, 'recon')
        result_save_path_gt = os.path.join(result_save_path, 'gt')
        result_save_path_mask = os.path.join(result_save_path, 'mask')
        result_save_path_masked = os.path.join(result_save_path, 'masked')
        result_save_path_fake = os.path.join(result_save_path, 'fake')
        result_degree_save_path = os.path.join(result_save_path, 'degree')
        result_degree_save_path_gt = os.path.join(result_degree_save_path, 'gt')
        result_degree_save_path_mask = os.path.join(result_degree_save_path, 'mask')
        result_degree_save_path_recon = os.path.join(result_degree_save_path, 'recon')

        for d in [result_save_path_recon, result_save_path_gt, result_save_path_mask,
                result_save_path_masked, result_save_path_fake, result_degree_save_path,
                result_degree_save_path_gt, result_degree_save_path_mask, result_degree_save_path_recon]:
            os.makedirs(d, exist_ok=True)

        pbar = tqdm(total=total_test_images, desc="Processing test images")

        with torch.no_grad():
            for batch_idx, items in enumerate(test_loader):
                try:
                    if len(items) == 2:
                        gt_images, masks = self.__cuda__(*items)
                        batch_size = gt_images.size(0)
                        filenames = [f"test_{count + i + 1}" for i in range(batch_size)]
                    elif len(items) == 3:
                        gt_images, masks, filenames = self.__cuda__(*items)
                        batch_size = gt_images.size(0)
                        if isinstance(filenames, torch.Tensor):
                            filenames = [f"test_{count + i + 1}" for i in range(batch_size)]
                    else:
                        raise ValueError(f"Expected 2 or 3 items, but got {len(items)}")

                    # ===== 테스트용 전처리 (훈련과 동일) =====
                    gt_processed, gt_valid_mask = self.preprocess_rrs_data(
                        gt_images, mode='adaptive_fill'
                    )

                    # 마스크 이진화
                    masks = (masks > 0).float()

                    # 정규화 (훈련과 동일)
                    gt_normalized, data_min, data_max = self.smart_normalize(gt_processed, gt_valid_mask)
                    masked_images = gt_normalized * masks

                    # ===== 입력 데이터 NaN/Inf 체크 =====
                    if (torch.isnan(gt_normalized).any() or torch.isinf(gt_normalized).any() or
                        torch.isnan(masked_images).any() or torch.isinf(masked_images).any()):
                        print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected after preprocessing")
                        skipped_count += batch_size
                        pbar.update(batch_size)
                        continue

                    # 첫 번째 배치에서 데이터 범위 확인
                    if batch_idx == 0:
                        print(f"\n=== Test Data Statistics ===")
                        print(f"Original range: [{gt_images.min():.3f}, {gt_images.max():.3f}]")
                        print(f"Processed range: [{gt_processed.min():.3f}, {gt_processed.max():.3f}]")
                        print(f"Normalized range: [{gt_normalized.min():.3f}, {gt_normalized.max():.3f}]")
                        print(f"Normalization params: min={data_min:.2f}, max={data_max:.2f}")

                    # Generator 시도
                    try:
                        # forward pass
                        masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_normalized)

                        # Clamp the output to be within [0, 1] range
                        fake_B = torch.clamp(fake_B, 0, 1)

                        # DEBUG: Raw model output
                        print(f"  - DEBUG: Raw model output (normalized space) min: {torch.min(fake_B):.4f}, max: {torch.max(fake_B):.4f}")

                        # 정규화된 결과를 원본 스케일로 복원
                        fake_B_original = self.denormalize_to_original(fake_B, data_min, data_max)
                        comp_B_original = self.denormalize_to_original(comp_B, data_min, data_max)

                        # ===== Generator 출력 NaN/Inf 체크 =====
                        if (torch.isnan(fake_B_original).any() or torch.isinf(fake_B_original).any() or
                            torch.isnan(comp_B_original).any() or torch.isinf(comp_B_original).any()):
                            print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in generator output")
                            skipped_count += batch_size
                            pbar.update(batch_size)
                            continue

                        generator_success_count += 1

                    except Exception as e:
                        print(f"Generator failed on batch {batch_idx}: {e}")
                        # Fallback
                        try:
                            fake_B_normalized = self.improved_inpainting_fallback(masked_images, masks)
                            comp_B_normalized = fake_B_normalized * (1 - masks) + gt_normalized * masks

                            # 원본 스케일로 복원
                            fake_B_original = self.denormalize_to_original(fake_B_normalized, data_min, data_max)
                            comp_B_original = self.denormalize_to_original(comp_B_normalized, data_min, data_max)

                            if (torch.isnan(fake_B_original).any() or torch.isinf(fake_B_original).any() or
                                torch.isnan(comp_B_original).any() or torch.isinf(comp_B_original).any()):
                                print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in fallback output")
                                skipped_count += batch_size
                                pbar.update(batch_size)
                                continue

                            fake_B = fake_B_normalized  # 저장용
                            comp_B = comp_B_normalized
                            masked_image = masked_images
                            fallback_count += 1

                        except Exception as fallback_error:
                            print(f"❌ Fallback also failed on batch {batch_idx}: {fallback_error}")
                            skipped_count += batch_size
                            pbar.update(batch_size)
                            continue

                    # 첫 번째 배치에서 출력 범위 확인
                    if batch_idx == 0:
                        print(f"Output range - Original scale: [{fake_B_original.min():.3f}, {fake_B_original.max():.3f}]")
                        print("=" * 50)

                    # 배치 내 각 샘플별 결과 저장
                    for k in range(batch_size):
                        # ===== 개별 샘플 NaN/Inf 체크 =====
                        sample_gt_norm = gt_normalized[k:k+1]
                        sample_gt_orig = gt_processed[k:k+1]  # 원본 스케일 GT
                        sample_masked = masked_image[k:k+1]
                        sample_fake = fake_B[k:k+1]
                        sample_comp = comp_B[k:k+1]
                        sample_mask = masks[k:k+1]

                        # 각 샘플에 대해 NaN/Inf 체크
                        if (torch.isnan(sample_gt_norm).any() or torch.isinf(sample_gt_norm).any() or
                            torch.isnan(sample_masked).any() or torch.isinf(sample_masked).any() or
                            torch.isnan(sample_fake).any() or torch.isinf(sample_fake).any() or
                            torch.isnan(sample_comp).any() or torch.isinf(sample_comp).any() or
                            torch.isnan(sample_mask).any() or torch.isinf(sample_mask).any()):
                            print(f"❌ Skipping sample {count + 1}: NaN/Inf detected in individual sample")
                            skipped_count += 1
                            pbar.update(1)
                            continue

                        count += 1

                        # filename 처리
                        if isinstance(filenames[k], str):
                            filename = filenames[k]
                        else:
                            filename = f"test_{count}"
                        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]

                        # 저장할 파일 경로 정의
                        gt_file_prefix = os.path.join(result_save_path_gt, f"gt_{count}_{filename_no_ext}.png")
                        mask_file_prefix = os.path.join(result_save_path_mask, f"mask_{count}_{filename_no_ext}.png")
                        masked_file_prefix = os.path.join(result_save_path_masked, f"masked_{count}_{filename_no_ext}.png")
                        fake_file_prefix = os.path.join(result_save_path_fake, f"fake_{count}_{filename_no_ext}.png")
                        recon_file_prefix = os.path.join(result_save_path_recon, f"recon_{count}_{filename_no_ext}.png")

                        try:
                            # 결과 이미지를 그리드 형식으로 저장 (정규화된 버전)
                            self.save_batch_images_grid(sample_gt_norm, gt_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_masked, masked_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_fake, fake_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_comp, recon_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_mask, mask_file_prefix, nrow=1, normalize=False)

                            # CSV 저장용 데이터 준비 (원본 스케일)
                            if sample_comp.size(1) > 1:
                                # 다중 채널인 경우 - 채널 평균 사용
                                fake_degree = self.denormalize_to_original(sample_fake[0].mean(dim=0), data_min, data_max).cpu().numpy()
                                gt_degree = sample_gt_orig[0, 1, :, :].cpu().numpy()  # 두 번째 채널
                                mask_degree = sample_mask[0, 1, :, :].cpu().numpy()
                            else:
                                # 단일 채널인 경우
                                fake_degree = self.denormalize_to_original(sample_fake[0, 0], data_min, data_max).cpu().numpy()
                                gt_degree = sample_gt_orig[0, 0].cpu().numpy()
                                mask_degree = sample_mask[0, 0].cpu().numpy()

                            # ===== CSV 데이터 NaN/Inf 체크 =====
                            if (np.isnan(fake_degree).any() or np.isinf(fake_degree).any() or
                                np.isnan(gt_degree).any() or np.isinf(gt_degree).any() or
                                np.isnan(mask_degree).any() or np.isinf(mask_degree).any()):
                                print(f"❌ Skipping sample {count} CSV save: NaN/Inf detected in numpy arrays")
                                # 이미 저장된 이미지 파일들을 삭제
                                for file_path in [gt_file_prefix, mask_file_prefix, masked_file_prefix,
                                                fake_file_prefix, recon_file_prefix]:
                                    if os.path.exists(file_path):
                                        os.remove(file_path)
                                count -= 1  # 카운트 롤백
                                skipped_count += 1
                                continue

                            # 육지/해양 구분 처리 (원본 스케일 기준)
                            if sample_gt_orig.size(1) > 1:
                                gt_channel = sample_gt_orig[0, 1, :, :].cpu().numpy()
                            else:
                                gt_channel = sample_gt_orig[0, 0, :, :].cpu().numpy()

                            # 해양 영역: -999가 아닌 값들
                            ocean_region = (gt_channel != self.missing_value)

                            # 육지 영역은 결측치 계산에서 제외
                            mask_degree[~ocean_region] = 1

                            # 육지 픽셀을 255로 마킹 (validation 함수와 호환)
                            land_mask = ~ocean_region
                            gt_degree[land_mask] = 255
                            fake_degree[land_mask] = 255

                            # CSV 저장
                            np.savetxt(os.path.join(result_degree_save_path_recon, f"img_{count}_{filename_no_ext}.csv"),
                                    fake_degree, delimiter=",", fmt='%.6e')
                            np.savetxt(os.path.join(result_degree_save_path_gt, f"gt_{count}_{filename_no_ext}.csv"),
                                    gt_degree, delimiter=",", fmt='%.6e')
                            np.savetxt(os.path.join(result_degree_save_path_mask, f"mask_{count}_{filename_no_ext}.csv"),
                                    mask_degree, delimiter=",", fmt='%.6f')

                        except Exception as e:
                            print(f"❌ Error saving sample {count}: {e}")
                            # 저장 실패 시 이미 저장된 파일들 삭제
                            for file_path in [gt_file_prefix, mask_file_prefix, masked_file_prefix,
                                            fake_file_prefix, recon_file_prefix]:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            count -= 1  # 카운트 롤백
                            skipped_count += 1
                            continue

                        # 개별 샘플 처리 후 진행바 업데이트
                        pbar.update(1)

                    ei_time = time.time()
                    elapsed_time = ei_time - s_time

                    # 진행바에 경과 시간 및 현재 진행률을 postfix로 표시
                    percent = (count / total_test_images) * 100 if total_test_images > 0 else 0
                    pbar.set_postfix({
                        "processed": count,
                        "skipped": skipped_count,
                        "elapsed": f"{elapsed_time:.2f}s",
                        "percent": f"{percent:.2f}%"
                    })

                except Exception as e:
                    print(f"❌ Error processing batch {batch_idx}: {e}")
                    # 배치 전체 스킵
                    if 'batch_size' in locals():
                        skipped_count += batch_size
                        pbar.update(batch_size)
                    continue

        pbar.close()
        e_time = time.time()
        total_time = e_time - s_time

        # 결과 요약
        print(f"\n=== Test Completed ===")
        print(f"Data processing: RRS-aware preprocessing with original scale output")
        print(f"Total samples processed: {count}")
        print(f"Total samples skipped: {skipped_count}")
        print(f"Success rate: {count/(count+skipped_count)*100:.1f}%" if (count+skipped_count) > 0 else "No samples")
        print(f"Generator success: {generator_success_count}")
        print(f"Fallback used: {fallback_count}")
        if generator_success_count + fallback_count > 0:
            print(f"Generator success rate: {generator_success_count/(generator_success_count+fallback_count)*100:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per sample: {total_time/count:.3f}s" if count > 0 else "No samples processed")
        print(f"Results saved to: {result_save_path}")
        print(f"CSV files saved to: {result_degree_save_path}")
        print("CSV files contain original scale values for validation")

        for para in self.G.parameters():
            para.requires_grad = True

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        return masked_image, self.fake_B, self.comp_B

    def improved_inpainting_fallback(self, masked_image, mask):
        """
        Improved fallback inpainting using better interpolation
        """
        import torch.nn.functional as F
        fake_B = masked_image.clone()

        for i in range(fake_B.size(0)):  # batch
            for c in range(fake_B.size(1)):  # channel
                img_slice = fake_B[i, c]
                mask_slice = mask[i, c]

                # 구멍 영역 찾기
                hole_mask = (mask_slice == 0)
                if hole_mask.sum() > 0:
                    # 가우시안 블러를 이용한 더 자연스러운 interpolation
                    kernel_size = 5
                    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(img_slice.device) / (kernel_size * kernel_size)
                    # 패딩을 추가하여 경계 효과 방지
                    padding = kernel_size // 2
                    blurred = F.conv2d(
                        img_slice.unsqueeze(0).unsqueeze(0),
                        kernel,
                        padding=padding
                    ).squeeze()

                    # 구멍 부분만 블러된 값으로 채우기
                    img_slice[hole_mask] = blurred[hole_mask]

                    # 추가로 주변 픽셀의 평균으로 보정
                    valid_pixels = img_slice[mask_slice == 1]
                    if len(valid_pixels) > 0:
                        mean_val = valid_pixels.mean()
                        # 블러된 값과 평균값의 가중평균
                        img_slice[hole_mask] = 0.7 * img_slice[hole_mask] + 0.3 * mean_val
        return fake_B

    def denormalize_to_original(self, normalized_tensor, data_min, data_max):
        """
        정규화된 텐서를 원본 데이터 범위로 복원
        """
        return normalized_tensor * (data_max - data_min) + data_min

    def check_model_health(self):
        """
        Check if the generator model is in a healthy state
        """
        print("Checking model parameters...")
        # 모델 파라미터 체크
        total_params = 0
        nan_params = 0
        inf_params = 0
        zero_params = 0

        for name, param in self.G.named_parameters():
            total_params += param.numel()
            if torch.isnan(param).any():
                nan_params += torch.isnan(param).sum().item()
                print(f"❌ NaN found in parameter: {name}")
            if torch.isinf(param).any():
                inf_params += torch.isinf(param).sum().item()
                print(f"❌ Inf found in parameter: {name}")
            if torch.all(param == 0):
                zero_params += param.numel()
                print(f"⚠️  All zero parameter: {name}")

        print(f"Total parameters: {total_params:,}")
        print(f"NaN parameters: {nan_params}")
        print(f"Inf parameters: {inf_params}")
        print(f"Zero parameters: {zero_params}")

        if nan_params > 0 or inf_params > 0:
            print("❌ Model has corrupted parameters!")
            return False
        elif zero_params > total_params * 0.5:
            print("⚠️  Model might not be properly trained (too many zero weights)")
            return False
        else:
            print("✅ Model parameters look healthy")
            return True

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