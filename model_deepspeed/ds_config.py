# model_deepspeed/ds_config.py

ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "fp16": {
        "enabled": True,
        # 필요에 따라 loss scaling 관련 추가 옵션들(예: 'loss_scale', 'loss_scale_window')을
        # 사용할 수도 있습니다.
        "initial_scale_power": 16,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        # CPU 오프로딩 옵션 (Stage 2에서는 optimizer만 가능)
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },

        # 기존에 있던 설정들
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "gradient_clipping": 1.0,
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True
    }
}
