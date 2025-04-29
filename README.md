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

- ust21에 값의 범위를 확인하니 0.01 ~ 10 사이의 값을 사용 권고 




 - 최종 목표는 ust21이 제공하는 영역이 목표입니다.




