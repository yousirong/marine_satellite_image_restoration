# scripts/inference.sh

#!/bin/bash

# 설정 변수
CONFIG_PATH="configs/ds_config.py"  # DeepSpeed 설정은 Python에서 관리하므로 필요 없음
CONFIG_FILE="config.yaml"  # YAML 설정 파일이 필요하다면 유지, 아니면 제거

# 기타 설정
DATA_ROOT="data/test/"
MASK_ROOT="data/masks/"
LAND_SEA_MASK_PATH="preprocessing/Land_mask/Land_mask.npy"
MASK_MODE="test_mode"
TARGET_SIZE=256
BATCH_SIZE=32
GPU_IDS=(0)
MASK_REVERSE=false
MASK_PATH="data/masks/"
RESULT_SAVE_PATH="performance/inference_results/"

# 추론 명령 실행
python scripts/inference.py \
    --model_save_path "checkpoints/" \
    --data_root ${DATA_ROOT} \
    --mask_root ${MASK_ROOT} \
    --land_sea_mask_path ${LAND_SEA_MASK_PATH} \
    --mask_mode ${MASK_MODE} \
    --target_size ${TARGET_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --gpu_ids ${GPU_IDS[@]} \
    --mask_reverse ${MASK_REVERSE} \
    --mask_path ${MASK_PATH} \
    --test \
    --result_save_path ${RESULT_SAVE_PATH}
