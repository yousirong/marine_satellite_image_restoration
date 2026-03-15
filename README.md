# marine_satellite_image_restoration
프로젝트명: marine_satellite_image_restoration 해양위성영상 분석 및 활용 연구

```bash
GOCI-UST21 통합 복원 테스트 베드가 완성되었습니다.

생성된 디렉터리 구조

/home/juneyonglee/Desktop/AY_ust/testbed/
├── __init__.py
├── configs/
│   ├── __init__.py
│   └── testbed.yaml              # 통합 설정 파일
├── core/
│   ├── __init__.py
│   ├── unified_model.py          # 통합 모델 래퍼
│   ├── base_dataset.py           # 데이터셋 베이스 + 팩토리
│   └── evaluator.py              # 평가기 베이스 + 팩토리
├── datasets/
│   ├── __init__.py
│   ├── goci_dataset.py           # GOCI Full Patch 데이터셋
│   └── ust21_dataset.py          # UST21 Full Patch 데이터셋
├── validators/
│   ├── __init__.py
│   ├── goci_validator.py         # GOCI 검증기
│   └── ust21_validator.py        # UST21 검증기
├── utils/
│   ├── __init__.py
│   ├── mask_utils.py             # 마스크 로더/변환 유틸
│   ├── metrics.py                # 성능 지표 (RMSE, MAE, R²)
│   └── visualization.py          # 시각화 도구
└── run_testbed.py                # 메인 실행 스크립트
실행 방법

cd /home/juneyonglee/Desktop/AY_ust

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

# 특정 날짜들만
python -m testbed.run_testbed --dates 20210101 20210115 20210131
주요 특징
기존 코드 재사용: model/model.py의 RFRNetModel을 직접 import하여 사용
통합 인터페이스: GOCI/UST21 구분 없이 동일한 방식으로 테스트
설정 파일 기반: testbed.yaml에서 모든 파라미터 관리
자동 검증: 추론 후 자동으로 RMSE, MAE, R² 계산 및 시각화
Full Patch 전용: 낙동강, 새만금 관련 코드 제외
```


```bash
UST21 
UST21 정확한 위도/경도 정보
NetCDF 파일에서 추출한 정확한 좌표 정보:
이미지 경계 좌표
좌상단 (Upper Left):
위도: 23.17°N
경도: 150.855°E
우하단 (Lower Right):
위도: 49.1579°N
경도: 111.604°E
전체 범위
위도 범위: 23.17°N ~ 49.1579°N (약 26도)
경도 범위: 111.604°E ~ 150.855°E (약 39.25도)
공간 해상도
위도 해상도: 0.03° (약 3.3 km)
경도 해상도: 0.03° (약 3.3 km at equator)
좌표계 정보
Datum: WGS84
Geographic CRS: WGS 1984 / UTM

GOCI 정확한 위도/경도 정보
HDF5 파일에서 추출한 정확한 좌표 정보:
투영 방식
투영법: Lambert Conformal Conic (LCC)
중심 위도: 36.0°N
중심 경도: 130.0°E
이미지 경계 좌표 (Geographic Coordinates)
좌상단 (Upper Left):
위도: 54.527°N
경도: 87.909°E
우상단 (Upper Right):
위도: 54.527°N
경도: 172.091°E
좌하단 (Lower Left):
위도: 7.125°N
경도: 107.299°E
우하단 (Lower Right):
위도: 7.125°N
경도: 152.701°E
전체 범위
위도 범위: 7.125°N ~ 54.527°N (약 47.4도)
경도 범위: 87.909°E ~ 172.091°E (약 84.2도)
이미지 크기 및 해상도
이미지 크기: 5685 rows × 5567 columns
총 픽셀 수: 31,648,395 픽셀
공간 해상도: 약 1.0 km/pixel (투영 좌표 기준)
투영 좌표 범위 (Projection Coordinates)
X 범위: -2783 km ~ 2783 km
Y 범위: -2986 km ~ 2698 km
비교
UST21: 23.17°N ~ 49.16°N, 111.60°E ~ 150.86°E (약 250m 해상도)
GOCI: 7.125°N ~ 54.527°N, 87.909°E ~ 172.091°E (약 1km 해상도)
```


```bash
### GOCI
python -m model.run --c configs/train.yaml
python /home/juneyonglee/Desktop/AY_ust/model/eval/eval_goci_fullpatch.py

python -m model.run --c configs/val.yaml --val
1. 날짜 범위로 처리

python /home/juneyonglee/Desktop/AY_ust/model/eval/eval_goci_fullpatch.py

python /home/juneyonglee/Desktop/AY_ust/performance/val_goci_fullpatch.py


python3 /home/juneyonglee/Desktop/AY_ust/differencemap/Differencemap_OC3GOCIvsUST21.py \
    --oc3_dir /home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/oc3_batch_results_daily \
    --khoa_dir /home/juneyonglee/Desktop/AY_ust/My_Book/UST21/01_day/2021/01 \
    --output_dir /home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_differencemap_results \
    --goci_land_mask /home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy \
    --ust_land_mask /home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy \
    --start_date 20210101 \
    --end_date 20210131


### UST21

python3 performance/val_ust21_fullpatch.py \
    --base_results_dir /home/juneyonglee/Desktop/AY_ust/myhdd/UST21/test/2020 \
    --base_performance_dir /home/juneyonglee/Desktop/AY_ust/myhdd/UST21/performance/2020 \
    --land_mask_path /home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat \
    --start_date 20201201 \
    --end_date 20201231

python3 differencemap/Differencemap_UST21vsModis.py \
      --ust21_perf_dir /home/juneyonglee/Desktop/AY_ust/myhdd/UST21/performance/2020 \
      --modis_dir /home/juneyonglee/Desktop/AY_ust/My_Book/MODIS/MODIS_aqua_8days \
      --output_dir /home/juneyonglee/Desktop/AY_ust/differencemap/results \
      --ust_land_mask /home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy \
      --verbose

```