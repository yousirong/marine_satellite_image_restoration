import argparse
import os
from model import RFRNetModel
# from model_withvalidation import RFRNetModel_withvalidation
from model.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.io import load_yaml, build_save_folder
import torch
# GOCI 1day new validate function
from model.val_chl import validate
# ust21 8day new validate function
# from model.val_chl_new import validate

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default="", type=str, help="config file path")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation with limited samples for preview')
    parser.add_argument('--result_save_path', type=str, default=None,
                        help='Path where the test results will be saved')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data_path from YAML')
    args = parser.parse_args()

    assert args.c != "", "Please provide config file (.yaml)"

    # 1) YAML 로드
    load_yaml(args, args.c)

    # 2) 커맨드라인 인자로 --result_save_path가 전달되면, YAML의 result_save_path_default를 덮어씀
    if args.result_save_path is not None:
        args.result_save_path_default = args.result_save_path

    # 3) 커맨드라인 인자로 --data_path가 전달되면, YAML의 data_path를 덮어씀
    if args.data_path is not None:
        print(f"Overriding data_path from YAML: {getattr(args, 'data_path', 'Not set')} -> {args.data_path}")
        args.data_path = args.data_path

    # 4) build_save_folder 호출
    args = build_save_folder(args)

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))

    model = RFRNetModel()

    # -------------------- TEST --------------------
    if args.test:
        # 최종적으로 result_save_path_default를 실제 사용
        result_save_path = args.result_save_path_default
        print("Final Test results will be saved at:", result_save_path)

        model.initialize_model(args.model_path, False, None, args.gpu_ids)
        model.cuda()

        # 데이터로더 구성
        dataloader = DataLoader(
            Dataset(
                args.data_root,
                args.mask_root,
                args.land_sea_mask_path,
                args.mask_mode,
                args.target_size,
                mask_reverse=False,
                training=False
            ),
            batch_size=args.batch_size,
            num_workers=args.n_threads
        )

        # 모델 테스트
        model.test(dataloader, result_save_path)

    # -------------------- VALIDATION --------------------
    elif args.val:
        # Quick 모드 체크
        if args.quick:
            print("=== Running Quick Validation (Limited Samples) ===")
            sample_size = 1000  # Quick 모드에서는 1000개 파일만 처리
            loss_rate_suffix = "quick_test"
        else:
            print("=== Running Full Validation ===")
            sample_size = getattr(args, 'sample_size', None)  # YAML에서 설정한 값 사용
            loss_rate_suffix = "full_validation"

        print(f"Using data_path: {args.data_path}")
        print(f"Using save_path: {args.save_path}")
        print(f"Using land_sea_mask_path: {args.land_sea_mask_path}")
        print(f"Using sample_size: {sample_size}")

        # loss_rate는 더미값 전달 (실제로는 마스크에서 계산됨)
        dummy_loss_rate = loss_rate_suffix

        try:
            validate(
                loss_rate=dummy_loss_rate,
                data_path=args.data_path,
                save_path=args.save_path,
                land_sea_mask_path=args.land_sea_mask_path,
                sample_size=sample_size  # Quick 모드일 때 100개로 제한
            )
        except Exception as e:
            print(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()

    # -------------------- TRAIN --------------------
    else:
        # 학습 시에는 model_save_path와 기타 인자를 사용
        model.initialize_model(args.model_path, True, args.model_save_path, args.gpu_ids)
        model.cuda()

        dataloader = DataLoader(
            Dataset(
                args.data_root,
                args.mask_root,
                args.land_sea_mask_path,
                args.mask_mode,
                args.target_size,
                mask_reverse=False
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_threads
        )

        model.train(dataloader, args.model_save_path, args.save_capacity, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()