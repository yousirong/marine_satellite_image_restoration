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

    # def save_batch_images_grid(self, images, save_path, nrow=8, padding=2, normalize=True, is_mask=False):
    #     if not isinstance(images, torch.Tensor):
    #         raise ValueError(f"Expected torch.Tensor but got {type(images)}")
    #     grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     if not save_path.endswith('.png'):
    #         save_path += '.png'
    #     save_image(grid, save_path)
    #     print(f"Image grid saved at {save_path}")

    def save_batch_images_grid(self, images, save_path, nrow=8, padding=2, normalize=True, is_mask=False):
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor but got {type(images)}")

        # print(f"Saving image to {save_path}")
        # print(f"  Input tensor shape: {images.shape}")
        # print(f"  Input tensor dtype: {images.dtype}")
        # print(f"  Input tensor device: {images.device}")
        # print(f"  Input tensor range: [{images.min():.3f}, {images.max():.3f}]")
        # print(f"  Normalize: {normalize}")

        # NaN/Inf 체크
        if torch.isnan(images).any():
            print("  WARNING: Input contains NaN values!")
        if torch.isinf(images).any():
            print("  WARNING: Input contains Inf values!")

        # 값이 모두 0인지 체크
        if torch.all(images == 0):
            print("  WARNING: All values are zero!")

        # 값 분포 체크
        # unique_vals = torch.unique(images)
        # if len(unique_vals) <= 5:
        #     print(f"  Unique values: {unique_vals}")

        try:
            grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
            # print(f"  Grid created successfully, shape: {grid.shape}, range: [{grid.min():.3f}, {grid.max():.3f}]")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not save_path.endswith('.png'):
                save_path += '.png'

            save_image(grid, save_path)
            # print(f"  Image saved successfully at {save_path}")

            # 저장된 파일 크기 체크
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                # print(f"  Saved file size: {file_size} bytes")

        except Exception as e:
            print(f"  Error during save: {e}")
            import traceback
            traceback.print_exc()
            raise e

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

                if self.iter % 100 == 0:
                    e_time = time.time()
                    print(f"Iter:{self.iter}, l1_loss:{self.l1_loss_val / 50:.4f}, time:{e_time - s_time:.2f}")
                    self.writer.add_scalar("Train/Loss", self.l1_loss_val, self.iter)
                    s_time = time.time()
                    self.l1_loss_val = 0.0

                if self.iter % 100 == 0:
                    prefix = os.path.join(save_path, 'training', f"{os.path.basename(save_path)}_{self.iter}")
                    self.save_batch_images_grid(gt_images, f"{prefix}_gt")
                    self.save_batch_images_grid(comp_B, f"{prefix}_img")
                    self.save_batch_images_grid(masked_images, f"{prefix}_masked")
                    self.save_batch_images_grid(masks, f"{prefix}_masks")
                    if is_available_to_store(store_capacity):
                        save_ckpt(f"{save_path}/g_{self.iter}.pth",
                                  [('generator', self.G)],
                                  [('optimizer_G', self.optm_G)],
                                  self.iter)

    def test(self, test_loader, result_save_path):
        """
        Test the RFRNet model on GOCI RRS dataset with CSV output for validation
        NaN/Inf 값이 발견되면 해당 샘플을 완전히 제외하고 저장하지 않음
        """
        # 먼저 모델 상태 체크
        print("=== Model Health Check ===")
        model_healthy = self.check_model_health()

        self.G.eval()  # 모델 평가 모드로 전환
        for para in self.G.parameters():
            para.requires_grad = False  # 테스트 시 gradient 계산 비활성화

        # 전달받은 result_save_path를 절대 경로로 변환하여 사용
        result_save_path = os.path.abspath(result_save_path)
        print("Saving test results to:", result_save_path)

        count = 0
        skipped_count = 0
        generator_success_count = 0
        fallback_count = 0
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
        result_save_path_fake = os.path.join(result_save_path, 'fake')
        result_degree_save_path = os.path.join(result_save_path, 'degree')
        result_degree_save_path_gt = os.path.join(result_degree_save_path, 'gt')
        result_degree_save_path_mask = os.path.join(result_degree_save_path, 'mask')
        result_degree_save_path_recon = os.path.join(result_degree_save_path, 'recon')

        for d in [result_save_path_recon, result_save_path_gt, result_save_path_mask,
                result_save_path_masked, result_save_path_fake, result_degree_save_path,
                result_degree_save_path_gt, result_degree_save_path_mask, result_degree_save_path_recon]:
            os.makedirs(d, exist_ok=True)

        # tqdm 진행바 생성: 전체 테스트 이미지 개수를 총합으로 설정
        pbar = tqdm(total=total_test_images, desc="Processing test images")

        with torch.no_grad():
            # 테스트 데이터셋의 배치를 순회
            for batch_idx, items in enumerate(test_loader):
                try:
                    # items에 filename 정보가 없는 경우, 임의로 생성
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

                    # ===== 입력 데이터 NaN/Inf 체크 =====
                    if torch.isnan(gt_images).any() or torch.isinf(gt_images).any():
                        print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in gt_images")
                        skipped_count += batch_size
                        pbar.update(batch_size)
                        continue

                    if torch.isnan(masks).any() or torch.isinf(masks).any():
                        print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in masks")
                        skipped_count += batch_size
                        pbar.update(batch_size)
                        continue

                    si_time = time.time()

                    # 마스크를 0/1 float 텐서로 변환 (0: 육지, 1: 해양)
                    masks = (masks > 0).float()

                    # gt_images와 마스크를 곱하여 해양 영역만 추출
                    masked_images = gt_images * masks

                    # ===== 마스크 적용 후 NaN/Inf 체크 =====
                    if torch.isnan(masked_images).any() or torch.isinf(masked_images).any():
                        print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in masked_images")
                        skipped_count += batch_size
                        pbar.update(batch_size)
                        continue

                    # 첫 번째 배치에서 데이터 범위 확인
                    if batch_idx == 0:
                        print(f"\n=== Raw Data Statistics (Training Compatible) ===")
                        print(f"GT images: min={gt_images.min():.3f}, max={gt_images.max():.3f}, mean={gt_images.mean():.3f}")
                        print(f"Masks: min={masks.min():.3f}, max={masks.max():.3f}, mean={masks.mean():.3f}")
                        print(f"Masked images: min={masked_images.min():.3f}, max={masked_images.max():.3f}, mean={masked_images.mean():.3f}")
                        print("Using original RRS values (same as training)")

                    # Generator 시도
                    try:
                        # forward pass (train과 동일한 방식으로 처리)
                        masked_image, fake_B, comp_B = self.forward(masked_images, masks, gt_images)

                        # ===== Generator 출력 NaN/Inf 체크 =====
                        if torch.isnan(fake_B).any() or torch.isinf(fake_B).any():
                            print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in generator output")
                            skipped_count += batch_size
                            pbar.update(batch_size)
                            continue

                        if torch.isnan(comp_B).any() or torch.isinf(comp_B).any():
                            print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in composition output")
                            skipped_count += batch_size
                            pbar.update(batch_size)
                            continue

                        generator_success_count += 1

                    except Exception as e:
                        print(f"Generator failed on batch {batch_idx}: {e}")
                        # Fallback: 원본 값 기반 interpolation
                        try:
                            fake_B = self.improved_inpainting_fallback(masked_images, masks)
                            comp_B = fake_B * (1 - masks) + gt_images * masks
                            masked_image = masked_images

                            # ===== Fallback 출력 NaN/Inf 체크 =====
                            if torch.isnan(fake_B).any() or torch.isinf(fake_B).any():
                                print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in fallback output")
                                skipped_count += batch_size
                                pbar.update(batch_size)
                                continue

                            if torch.isnan(comp_B).any() or torch.isinf(comp_B).any():
                                print(f"❌ Skipping batch {batch_idx}: NaN/Inf detected in fallback composition")
                                skipped_count += batch_size
                                pbar.update(batch_size)
                                continue

                            fallback_count += 1

                        except Exception as fallback_error:
                            print(f"❌ Fallback also failed on batch {batch_idx}: {fallback_error}")
                            skipped_count += batch_size
                            pbar.update(batch_size)
                            continue

                    # 첫 번째 배치에서 출력 범위 확인
                    if batch_idx == 0:
                        print(f"Fake B: min={fake_B.min():.3f}, max={fake_B.max():.3f}, mean={fake_B.mean():.3f}")
                        print(f"Comp B: min={comp_B.min():.3f}, max={comp_B.max():.3f}, mean={comp_B.mean():.3f}")
                        print("=" * 50)

                    # 배치 내 각 샘플별 결과 저장
                    for k in range(batch_size):
                        # ===== 개별 샘플 NaN/Inf 체크 =====
                        sample_gt = gt_images[k:k+1]
                        sample_masked = masked_image[k:k+1]
                        sample_fake = fake_B[k:k+1]
                        sample_comp = comp_B[k:k+1]
                        sample_mask = masks[k:k+1]

                        # 각 샘플에 대해 NaN/Inf 체크
                        if (torch.isnan(sample_gt).any() or torch.isinf(sample_gt).any() or
                            torch.isnan(sample_masked).any() or torch.isinf(sample_masked).any() or
                            torch.isnan(sample_fake).any() or torch.isinf(sample_fake).any() or
                            torch.isnan(sample_comp).any() or torch.isinf(sample_comp).any() or
                            torch.isnan(sample_mask).any() or torch.isinf(sample_mask).any()):

                            print(f"❌ Skipping sample {count + 1}: NaN/Inf detected in individual sample")
                            skipped_count += 1
                            pbar.update(1)
                            continue

                        count += 1

                        # filename이 이미 있다면 사용, 없으면 임의의 이름을 사용
                        if isinstance(filenames[k], str):
                            filename = filenames[k]
                        else:
                            filename = f"test_{count}"
                        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]

                        # 저장할 파일 경로 정의 (PNG 형식으로 저장)
                        gt_file_prefix = os.path.join(result_save_path_gt, f"gt_{count}_{filename_no_ext}.png")
                        mask_file_prefix = os.path.join(result_save_path_mask, f"mask_{count}_{filename_no_ext}.png")
                        masked_file_prefix = os.path.join(result_save_path_masked, f"masked_{count}_{filename_no_ext}.png")
                        fake_file_prefix = os.path.join(result_save_path_fake, f"fake_{count}_{filename_no_ext}.png")
                        recon_file_prefix = os.path.join(result_save_path_recon, f"recon_{count}_{filename_no_ext}.png")

                        try:
                            # 결과 이미지를 그리드 형식으로 저장
                            self.save_batch_images_grid(sample_gt, gt_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_masked, masked_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_fake, fake_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_comp, recon_file_prefix, nrow=1, normalize=True)
                            self.save_batch_images_grid(sample_mask, mask_file_prefix, nrow=1, normalize=False)

                            # 채널별 평균(예: degree 계산) 데이터를 CSV로 저장
                            if sample_fake.size(1) > 1:
                                # 다중 채널인 경우
                                fake_degree = sample_fake[0].mean(dim=0).cpu().numpy()  # 채널 평균
                                gt_degree = sample_gt[0, 1, :, :].cpu().numpy()      # 두 번째 채널 (예시)
                                mask_degree = sample_mask[0, 1, :, :].cpu().numpy()
                            else:
                                # 단일 채널인 경우
                                fake_degree = sample_fake[0, 0].cpu().numpy()
                                gt_degree = sample_gt[0, 0].cpu().numpy()
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

                            # gt_images의 채널을 기준으로 육지와 해양을 구분 (해양: 값 > 0)
                            if sample_gt.size(1) > 1:
                                gt_channel = sample_gt[0, 1, :, :].cpu().numpy()
                            else:
                                gt_channel = sample_gt[0, 0, :, :].cpu().numpy()

                            ocean_region = (gt_channel > 0)
                            # 육지 영역(해양이 아닌 부분)은 결측치 계산에서 제외하기 위해 유효한 값(예: 1)으로 설정
                            mask_degree[~ocean_region] = 1

                            # 육지 픽셀을 255로 마킹 (validation 함수와 호환)
                            land_mask = (mask_degree == 0) & (gt_degree <= 0)
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
                    elapsed_time = ei_time - s_time  # 전체 경과 시간

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
        print(f"Data processing: Original RRS values (same as training)")
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
        Improved fallback inpainting using better interpolation (원본 스케일 유지)
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

    def prepare_mask_for_csv(self, mask_tensor, gt_tensor):
        """
        마스크를 CSV 저장용으로 준비 (validation 함수 호환)
        """
        # 마스크: 1=보존할 부분, 0=복원할 부분
        # validation에서는 육지를 255로 표현하므로 이에 맞춰 변환
        mask_for_csv = mask_tensor.clone()

        # GT가 육지 영역인 곳은 255로 설정 (validation 함수와 호환)
        # 이는 land_sea_mask에서 육지 부분을 의미

        return mask_for_csv


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
