# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    num_max_steps = 1000
    eval_interval = 10
    save_interval = 100
    path_dataset = "data/data_processed"
    model_name: str = "meta-llama/Llama-2-7b-hf"
    enable_fsdp: bool = True
    run_validation: bool = True
    batch_size_training: int = 4
    num_epochs: int = 1
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.1
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "samsum_dataset"
    micro_batch_size: int = 4
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "checkpoints"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = True
    save_model: bool = True
    dist_checkpoint_root_folder: str = "checkpoints"  # will be used if using FSDP
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = True  # will be used if using FSDP
    wandb: bool = True
    wandb_project: str = "llama-pretrain"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    checkpoint_folder: str = "checkpoints"
    verbose = True
