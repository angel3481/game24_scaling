# Import required libraries
from tsllm.rl.trainer.mcts_trainer_traj_ct2_value import AccelerateMCTSTrainer
from tsllm.rl.config import RLConfig
from peft import LoraConfig, PeftType
from tsllm.model.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# Replace standard attention with flash attention for better memory efficiency
replace_llama_attn_with_flash_attn()

# Configuration dictionary for model, training, and optimization
config = {
    "model": {
        "model_path": "meta-llama/Llama-2-7b-hf",  # Which model to use
    },
    "tokenizer": {
        "tokenizer_path": "meta-llama/Llama-2-7b-hf",  # Must match model path
        "padding_side": "right",  # Where to add padding tokens
    },
    "optimizer": {
        "name": "adamw",  # AdamW optimizer
        "kwargs": dict(lr=2.0e-5, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0),  # Optimizer settings
    },
    "scheduler": {
        "name": "cosine_warmup",  # Learning rate scheduler
        "kwargs": dict(warmup_ratio=0.03),  # Warm up first 3% of training
    },
    "train": {
        "pre_onpolicy_datapath": "../../tsllm/offline_rl/game24/processed/game24_train_cot_sample_offline_sft_k100_merged_dedup_sample17x3.jsonl",  # Training data
        "pre_onpolicy_datapath_train_test": "../../tsllm/offline_rl/game24/processed/game24_train_cot_sample_offline_sft_k100_ep3_dedup_sample17_train_test_sample_3.jsonl",  # Test data
        "env_name": "game24",  # Which environment/task to use
        "epochs": 3,  # Number of training epochs
        "train_epoch": 1,  # Current epoch
        "gamma": 1.0,  # Discount factor for future rewards
        "gae_lambda": 0.95,  # Lambda parameter for GAE
        "seq_length": 1024,  # Maximum sequence length
        "micro_batch_size": 4,  # Batch size per GPU
        "gradient_accumulation_steps": 4,  # Accumulate gradients over 4 batches
        "value_loss_coef": 1.0,  # Weight of the value loss
        "eval_interval": 1,  # Evaluate every epoch
        "checkpoint_interval": 1,  # Save model every epoch
        "checkpoint_dir": "checkpoints",  # Where to save models
        "save_optimizer": False,  # Don't save optimizer state
        "tracker": "tensorboard",  # Use TensorBoard for logging
        "logging_dir": "logs/",  # Where to save logs
        "project_name": "game24_scaling_llama2_7b",  # TODO: Change to game24_scaling_llama3_1b when switching to Llama-3-1b
        "onpolicy_per_problem_max_size": 1000,  # Maximum examples per problem type
    },
    "mcts": {},  # MCTS-specific settings (empty for now)
    "env": {},  # Environment-specific settings (using defaults)
}

# Convert dictionary to RLConfig object for validation
config = RLConfig.from_dict(config)

# Initialize trainer with validated config
trainer = AccelerateMCTSTrainer(config)

# Start training
trainer.learn()
