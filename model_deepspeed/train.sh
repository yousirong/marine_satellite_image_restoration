#!/bin/bash

# 설정 변수
MODEL_SAVE_PATH="/home/juneyonglee/MyData/5th_years/ust21_chl_8day"
DATA_ROOT="/media/juneyonglee/My Book/Preprocessed/UST21/train"
MASK_REVERSE=false
MASK_PATH="/media/juneyonglee/My Book/Preprocessed/UST21/mask/train"
LAND_SEA_MASK_PATH="preprocessing/Land_mask/Land_mask.npy"
MASK_MODE="train_mode"
TARGET_SIZE=256
BATCH_SIZE=32
LEARNING_RATE=5e-6
NUM_EPOCHS=150

SAVE_CAPACITY=5
NUM_WORKERS=16

# GPU_IDS 배열 정의 (여러 GPU 사용 시 공백으로 구분)
GPU_IDS=(0 1)  # 필요에 따라 GPU ID를 추가하거나 수정

# GPU_IDS 배열을 콤마로 구분된 문자열로 변환하여 CUDA_VISIBLE_DEVICES 설정
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")

# GPU의 수 계산
NUM_GPUS=${#GPU_IDS[@]}

# CUDA_VISIBLE_DEVICES 환경 변수 설정
export CUDA_VISIBLE_DEVICES

# PYTHONPATH 설정 (프로젝트 루트 디렉토리를 추가)
export PYTHONPATH=$PYTHONPATH:/home/juneyonglee/Desktop/AY_ust/

# 변수 출력 (디버깅용)
echo "MODEL_SAVE_PATH: ${MODEL_SAVE_PATH}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "MASK_ROOT: ${MASK_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Launch training with DeepSpeed
deepspeed --num_gpus=${NUM_GPUS} \
          model_deepspeed/train_deepspeed.py \
          --model_save_path "${MODEL_SAVE_PATH}" \
          --data_root "${DATA_ROOT}" \
          --mask_path "${MASK_PATH}" \
          --land_sea_mask_path "${LAND_SEA_MASK_PATH}" \
          --mask_mode "${MASK_MODE}" \
          --target_size "${TARGET_SIZE}" \
          --batch_size "${BATCH_SIZE}" \
          --learning_rate "${LEARNING_RATE}" \
          --num_epochs "${NUM_EPOCHS}" \
          --gpu_ids "${GPU_IDS[@]}" \
          --mask_reverse "${MASK_REVERSE}" \
          --save_capacity "${SAVE_CAPACITY}" \
          --num_workers "${NUM_WORKERS}" \
          --finetune  # 필요한 경우 추가 옵션
