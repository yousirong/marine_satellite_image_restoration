#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GOCI-UST21 Unified Restoration Testbed
======================================

GOCI와 UST21 모델을 통합하여 테스트하는 메인 스크립트입니다.

사용법:
    # 전체 테스트 (GOCI + UST21)
    python -m testbed.run_testbed --config testbed/configs/testbed.yaml

    # GOCI만 테스트
    python -m testbed.run_testbed --model goci

    # UST21만 테스트
    python -m testbed.run_testbed --model ust21

    # 날짜 범위 지정
    python -m testbed.run_testbed --model all --start 20210101 --end 20210131

    # GOCI 특정 밴드만
    python -m testbed.run_testbed --model goci --bands 2 4
"""

import argparse
import yaml
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader

# 프로젝트 루트 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from testbed.core.unified_model import UnifiedModel
from testbed.core.base_dataset import DatasetFactory
from testbed.core.evaluator import EvaluatorFactory
from testbed.utils.visualization import UnifiedVisualizer


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_date_range(start: str, end: str) -> List[str]:
    """날짜 범위 생성"""
    start_dt = datetime.strptime(start, '%Y%m%d')
    end_dt = datetime.strptime(end, '%Y%m%d')

    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return dates


def find_available_dates(base_path: str, pattern: str = 'goci') -> List[str]:
    """
    데이터 경로에서 사용 가능한 날짜 탐색

    Args:
        base_path: 베이스 경로
        pattern: 'goci' 또는 'ust21'

    Returns:
        List[str]: 날짜 문자열 목록 (YYYYMMDD)
    """
    dates = set()

    if not os.path.exists(base_path):
        print(f"[Warning] Path not found: {base_path}")
        return []

    if pattern == 'goci':
        # GOCI: band{N}/{year}/{date}/{time}/
        for year in os.listdir(base_path):
            year_path = os.path.join(base_path, year)
            if not os.path.isdir(year_path) or not year.isdigit():
                continue
            for date in os.listdir(year_path):
                date_path = os.path.join(year_path, date)
                if os.path.isdir(date_path) and len(date) == 8 and date.isdigit():
                    dates.add(date)
    else:
        # UST21: {year}/{date}/
        for year in os.listdir(base_path):
            year_path = os.path.join(base_path, year)
            if not os.path.isdir(year_path) or not year.isdigit():
                continue
            for date in os.listdir(year_path):
                date_path = os.path.join(year_path, date)
                if os.path.isdir(date_path) and len(date) == 8 and date.isdigit():
                    dates.add(date)

    return sorted(list(dates))


def run_goci_test(config: Dict[str, Any], dates: List[str],
                  bands: Optional[List[int]] = None):
    """
    GOCI 테스트 실행

    Args:
        config: 전체 설정
        dates: 테스트할 날짜 목록
        bands: 테스트할 밴드 목록 (선택적)
    """
    goci_config = config['models']['goci']
    common_config = config['common']

    if not goci_config.get('enabled', True):
        print("[GOCI] Disabled in config, skipping...")
        return

    bands = bands or goci_config.get('bands', [2, 3, 4])

    print(f"\n{'='*60}")
    print("GOCI RESTORATION TEST")
    print(f"{'='*60}")
    print(f"Bands: {bands}")
    print(f"Dates: {len(dates)} days")

    visualizer = UnifiedVisualizer({'colormap': goci_config['validation'].get('colormap', 'jet')})

    for band in bands:
        print(f"\n{'─'*40}")
        print(f"Processing Band {band}")
        print(f"{'─'*40}")

        # 체크포인트 확인
        checkpoint = goci_config['checkpoints'].get(f'band{band}')
        if not checkpoint or not os.path.exists(checkpoint):
            print(f"[WARN] Checkpoint not found for band{band}: {checkpoint}")
            continue

        # 모델 초기화
        model = UnifiedModel('goci', checkpoint, common_config['gpu_ids'])
        model.initialize()

        # 밴드별 데이터 경로
        band_base_path = os.path.join(goci_config['data']['base_path'], f'band{band}')

        # 날짜별 처리
        processed = 0
        for date_str in dates:
            year = date_str[:4]

            # 데이터 경로 확인
            data_path = os.path.join(band_base_path, year, date_str)
            if not os.path.exists(data_path):
                continue

            # 시간 폴더들 확인
            time_folders = [d for d in os.listdir(data_path)
                          if os.path.isdir(os.path.join(data_path, d))]
            if not time_folders:
                continue

            # 첫 번째 시간 폴더 사용 (또는 평균 처리 가능)
            time_folder = sorted(time_folders)[0]
            image_dir = os.path.join(data_path, time_folder)

            print(f"  Processing {date_str} ({time_folder})...")

            # 데이터셋 설정
            dataset_config = {
                'image_dir': image_dir,
                'land_mask_path': goci_config['data']['land_mask'],
                'patch_size': common_config['patch_size'],
                **goci_config.get('preprocessing', {})
            }

            # 데이터셋 생성
            try:
                dataset = DatasetFactory.create('goci', dataset_config)
            except Exception as e:
                print(f"    [Error] Failed to create dataset: {e}")
                continue

            if len(dataset) == 0:
                print(f"    [Skip] No files found")
                continue

            # DataLoader 생성
            dataloader = DataLoader(
                dataset,
                batch_size=common_config['batch_size'],
                num_workers=common_config['num_workers'],
                shuffle=False,
                pin_memory=True
            )

            # 결과 저장 경로
            result_path = os.path.join(
                common_config['output_base'],
                'goci', f'band{band}', year, date_str
            )
            os.makedirs(result_path, exist_ok=True)

            # 추론 실행
            try:
                model.infer(dataloader, result_path)
            except Exception as e:
                print(f"    [Error] Inference failed: {e}")
                continue

            # 검증
            try:
                evaluator = EvaluatorFactory.create('goci', goci_config['validation'])
                metrics = evaluator.evaluate(result_path)

                if 'error' not in metrics:
                    print(f"    RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}")

                    # 시각화 저장
                    scatter_path = os.path.join(result_path, 'scatter_plot.png')
                    gt, pred, _ = evaluator.load_results(result_path)
                    gt_filtered, pred_filtered = evaluator.filter_land(gt, pred)
                    if len(gt_filtered) > 0:
                        visualizer.plot_scatter(
                            gt_filtered, pred_filtered, metrics, scatter_path,
                            title=f"GOCI Band{band} - {date_str}"
                        )

                    # 요약 리포트 저장
                    report_path = os.path.join(result_path, 'report.txt')
                    visualizer.create_summary_report(metrics, report_path, f'goci_band{band}')

                    processed += 1
            except Exception as e:
                print(f"    [Error] Evaluation failed: {e}")

        print(f"\n  Band {band} completed: {processed}/{len(dates)} dates processed")


def run_ust21_test(config: Dict[str, Any], dates: List[str]):
    """
    UST21 테스트 실행

    Args:
        config: 전체 설정
        dates: 테스트할 날짜 목록
    """
    ust21_config = config['models']['ust21']
    common_config = config['common']

    if not ust21_config.get('enabled', True):
        print("[UST21] Disabled in config, skipping...")
        return

    print(f"\n{'='*60}")
    print("UST21 RESTORATION TEST")
    print(f"{'='*60}")
    print(f"Dates: {len(dates)} days")

    visualizer = UnifiedVisualizer({'colormap': ust21_config['validation'].get('colormap', 'jet')})

    # 체크포인트 확인
    checkpoint = ust21_config['checkpoints'].get('chl_8day')
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        return

    # 모델 초기화
    model = UnifiedModel('ust21', checkpoint, common_config['gpu_ids'])
    model.initialize()

    # 날짜별 처리
    processed = 0
    for date_str in dates:
        year = date_str[:4]

        # 데이터 경로 확인
        data_path = os.path.join(ust21_config['data']['base_path'], year, date_str)
        if not os.path.exists(data_path):
            continue

        print(f"  Processing {date_str}...")

        # 데이터셋 설정
        dataset_config = {
            'image_dir': data_path,
            'land_mask_path': ust21_config['data']['land_mask'],
            'patch_size': common_config['patch_size'],
            **ust21_config.get('preprocessing', {})
        }

        # 데이터셋 생성
        try:
            dataset = DatasetFactory.create('ust21', dataset_config)
        except Exception as e:
            print(f"    [Error] Failed to create dataset: {e}")
            continue

        if len(dataset) == 0:
            print(f"    [Skip] No files found")
            continue

        # DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=common_config['batch_size'],
            num_workers=common_config['num_workers'],
            shuffle=False,
            pin_memory=True
        )

        # 결과 저장 경로
        result_path = os.path.join(
            common_config['output_base'],
            'ust21', year, date_str
        )
        os.makedirs(result_path, exist_ok=True)

        # 추론 실행
        try:
            model.infer(dataloader, result_path)
        except Exception as e:
            print(f"    [Error] Inference failed: {e}")
            continue

        # 검증
        try:
            evaluator = EvaluatorFactory.create('ust21', ust21_config['validation'])
            metrics = evaluator.evaluate(result_path)

            if 'error' not in metrics:
                print(f"    RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}")

                # 시각화 저장
                scatter_path = os.path.join(result_path, 'scatter_plot.png')
                gt, pred, _ = evaluator.load_results(result_path)
                gt_filtered, pred_filtered = evaluator.filter_land(gt, pred)
                if len(gt_filtered) > 0:
                    visualizer.plot_scatter(
                        gt_filtered, pred_filtered, metrics, scatter_path,
                        title=f"UST21 Chl-a - {date_str}"
                    )

                # 요약 리포트 저장
                report_path = os.path.join(result_path, 'report.txt')
                visualizer.create_summary_report(metrics, report_path, 'ust21')

                processed += 1
        except Exception as e:
            print(f"    [Error] Evaluation failed: {e}")

    print(f"\n  UST21 completed: {processed}/{len(dates)} dates processed")


def main():
    parser = argparse.ArgumentParser(
        description='GOCI-UST21 Unified Restoration Testbed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m testbed.run_testbed --config testbed/configs/testbed.yaml
  python -m testbed.run_testbed --model goci --bands 2 4
  python -m testbed.run_testbed --model ust21 --start 20210101 --end 20210131
        """
    )
    parser.add_argument('--config', '-c', type=str,
                       default='testbed/configs/testbed.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', '-m', type=str,
                       choices=['goci', 'ust21', 'all'],
                       default='all',
                       help='Model to test (default: all)')
    parser.add_argument('--bands', '-b', type=int, nargs='+',
                       help='GOCI bands to test (for goci model)')
    parser.add_argument('--start', '-s', type=str,
                       help='Start date (YYYYMMDD)')
    parser.add_argument('--end', '-e', type=str,
                       help='End date (YYYYMMDD)')
    parser.add_argument('--dates', '-d', type=str, nargs='+',
                       help='Specific dates to process')

    args = parser.parse_args()

    # 설정 로드
    config_path = os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # 출력 디렉터리 생성
    output_base = config['common'].get('output_base', '/tmp/testbed_results')
    os.makedirs(output_base, exist_ok=True)

    # 날짜 결정
    if args.dates:
        dates = args.dates
    elif args.start and args.end:
        dates = generate_date_range(args.start, args.end)
    else:
        # 설정 파일의 기본값 사용
        test_scope = config.get('test_scope', {})
        date_range = test_scope.get('date_range', {})

        if date_range.get('start') and date_range.get('end'):
            dates = generate_date_range(date_range['start'], date_range['end'])
        else:
            # 자동 탐색
            print("[INFO] No date range specified, auto-detecting available dates...")

            if args.model in ['goci', 'all']:
                goci_base = config['models']['goci']['data']['base_path']
                bands = args.bands or config['models']['goci'].get('bands', [2])
                band_path = os.path.join(goci_base, f'band{bands[0]}')
                dates = find_available_dates(band_path, 'goci')[:7]  # 최근 7일
            else:
                ust21_base = config['models']['ust21']['data']['base_path']
                dates = find_available_dates(ust21_base, 'ust21')[:7]

            if not dates:
                print("[ERROR] No dates found. Please specify --start and --end")
                sys.exit(1)

    # 헤더 출력
    print(f"\n{'='*60}")
    print("GOCI-UST21 UNIFIED RESTORATION TESTBED")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Model: {args.model}")
    print(f"Output: {output_base}")
    if dates:
        print(f"Date range: {dates[0]} ~ {dates[-1]} ({len(dates)} days)")

    # 테스트 실행
    if args.model in ['goci', 'all']:
        run_goci_test(config, dates, args.bands)

    if args.model in ['ust21', 'all']:
        run_ust21_test(config, dates)

    print(f"\n{'='*60}")
    print("TESTBED COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_base}")


if __name__ == '__main__':
    main()
