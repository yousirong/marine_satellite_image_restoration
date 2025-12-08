# marine_satellite_image_restoration
프로젝트명: marine_satellite_image_restoration 해양위성영상 분석 및 활용 연구

주요 연구과제:

복원된 Rrs 데이터를 Chl-a로 변환 및 Difference Map 산출

목표: 복원된 Remote Sensing Reflectance (Rrs)를 Chlorophyll-a (Chl-a)로 변환 후, 합성장을 이용해 공간 분포의 오차를 시각화 (Difference Map).

주요 작업:
하루 8회 촬영된 GOCI 데이터를 픽셀 단위 평균을 통해 1일 데이터셋으로 변환.
새만금 또는 낙동강 지역의 특정 좌표를 선택해 복원된 Rrs 데이터를 Chl-a로 변환 후 결과 산출.
복원 결과와 실제 GOCI 데이터의 차이를 계산하여 시각화 (MinMaxScaler를 활용해 -20~+20 범위로 색상화).

성과:
현재까지 Rrs를 Chl-a로 변환해 성공적인 합성 결과를 얻은 사례가 드물지만, Difference Map을 통해 복원의 성능을 정량적으로 평가함.
Chl-a 합성장 공백 복원을 위한 지도 학습 모델 개발

목표: Chl-a 8일 평균 자료의 결손 영역을 복원하는 딥러닝 모델 개발.

주요 작업:
UST21 데이터셋을 8일 이동 평균으로 변환해 복원 모델의 훈련 자료로 사용.
예: 1월 1일1월 8일, 1월 2일1월 9일 식으로 이동 평균 계산.
연구 영역으로 새만금 및 낙동강 주변을 중심으로 설정하되, 데이터 부족 시 외해 쪽으로 영역 조정.
RFRNet 모델을 활용해 마스크를 씌운 데이터를 복원 훈련.

검증:
복원된 결과를 MODIS 8일 평균 자료와 비교해 성능 평가.
낙동강 및 새만금 지역의 복원 결과를 RMSE, MAE, R² 그래프로 정량적 분석.
기존의 256x256 패치 크기보다 큰 영역 테스트 이미지 복원 시, 데이터 간 공백 부분은 스무딩 기법 적용.

기술 스택 및 도구:

데이터 처리: Python, NumPy, PyTorch, TensorFlow
위성 데이터: GOCI Rrs, UST21 Chl-a, MODIS Chlorophyll-a
딥러닝 모델: RFRNet (Restoration Focused Refinement Network)
데이터 시각화 및 분석: Matplotlib, Seaborn, RMSE/MAE 평가 지표

성과:
복원된 Rrs 및 Chl-a 데이터를 통해 Difference Map 기반의 새로운 평가 기준 제안.
UST21 및 MODIS 데이터셋을 활용해 해양 위성영상 복원 기술 고도화.
실제 데이터에서 결손 영역 복원 성능을 극대화하여 연구 결과의 실질적 적용 가능성 제고.

연구 기여도:
해양위성 데이터의 활용성을 높이고, 복원 및 예측 모델의 성능을 분석하여 학계 및 산업적 응용 가능성을 확인.


2025년 5차년도
Rrs연구 복원, Chl-a 복원
    - Rrs 복원 : 천리안2B-LA(Local Area, 한반도 주변)영역을 대상으로 한 복원
    - Chl-a 복원 : 천리안2B-LA(Local Area, 한반도 주변)영역을 대상으로 한 복원

- ust21에 값의 범위를 확인하니 0.01 ~ 10 사이의 값을 사용하라고 권고 받았고 새로운 데이터를 업로드했으니 해당 위치의 데이터를 사용하는게 좋다고 의견 받았습니다
사용 우선순위가 높은 데이터는 UST21에서 5차년도에 새로 제공한 데이터입니다.
부득이하게 4차년도 데이터를 사용해야 하는 경우, 값의 범위를 0 ~ 20이 아니라 0.01 ~ 10으로 제한해야 합니다.

 -(위치) /home/data1/result/ust21(Chl_a) [이전에 제공했던 기간과 다릅니다. 확인 필요]


 - 최종 목표는 ust21이 제공하는 영역이 목표입니다.

 - 하지만, 현재 보유한 자원에 한계가 있어, 전 영역을 수행하기에는 어려움이 예상됩니다.

 - 따라서 연구 범위를 역으로 제안해 주셨으면 좋겠습니다. (단, ust21에서 제공하는 chl-a  생산 영역 전체를 커버할 수 있다면 전체 영역으로 진행)

 - 현재의 fetch 영역을 기준으로 몇 개의 실험 세트를 구성가능 한지, 서해/동해/남해 각각을 몇 개의 실험 구역으로 설정할 것인지, 우리나라 연안을 완전히 커버하기 위해 fetch를 어떻게 분할 해야 하는지 등을 검토해서 연구 영역을 도출하는게 좋다는게 제 의견입니다.

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