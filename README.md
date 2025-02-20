# Exploring Scaling Laws in Language Model Tasks: The Game of 24
The official implementation of paper: [Exploring Scaling Laws in Language Model Tasks: The Game of 24](TODO: Add paper link when available)

# Pre-trained Models
Currently using baseline models from the TS_LLM project (to be replaced with our own trained models):

* Game24:
	- Policy: https://huggingface.co/OhCherryFire/llama2-7b-game24-policy-hf
	- Value: https://huggingface.co/OhCherryFire/llama2-7b-game24-value

Note: These models serve as initial baselines. They will be replaced with our own trained models that specifically focus on studying scaling laws for the Game of 24 task.

# Environment Installation
Please use correct versions of `transformers` and `ctranslate2`:
```bash
conda create -n game24scaling python==3.10
conda activate game24scaling

pip install -r requirements.txt

pip install -e .
```


# Runnable Scripts

## Complete Training Pipeline

### Step 1: Policy Training (SFT)
Training scripts are located in `train_mcts_scripts/game24`:
```bash
cd train_mcts_scripts/game24

# Supervised Fine-Tuning (SFT) for Game of 24
accelerate launch --config_file mcts_game24_llama_deepspeed.yaml train_game24_sft.py 
# This creates checkpoint in: checkpoints/game24_scaling_llama2_7b_sft/checkpoint-3
```

### Step 2: Convert to CT2 Format
We use [Ctranslate2(3.17.1)](https://github.com/OpenNMT/CTranslate2) to speedup inference. After SFT training:
```bash
cd train_mcts_scripts/game24
mkdir ct2_models

# Convert the final SFT checkpoint to CT2 format
ct2-transformers-converter \
    --model checkpoints/game24_scaling_llama2_7b_sft/checkpoint-3 \
    --quantization bfloat16 \
    --output_dir ct2_models/llama2_sft_ep3_ct2
```
Note: We use bfloat16 for LLaMA models and float32 for GPT2 models.

### Step 3: Data Collection for Value Training
Using the CT2-converted model to generate training data:
```bash
cd tsllm/offline_rl

# Generate data using the CT2 model
sh game24/gen_3.sh ../train_mcts_scripts/game24/ct2_models "meta-llama/Llama-2-7b-hf"

# Process the generated data
sh game24/process.sh
```

### Step 4: Value Network Training
Train the value network (critic) using the collected data:
```bash
cd train_mcts_scripts/game24
accelerate launch --config_file mcts_game24_llama_deepspeed.yaml train_game24_critic.py
```

## Configuration Options
- You can customize training configurations in each Python file: `train_game24_sft.py`, `train_game24_critic.py`
- Additional parameters in `mcts_game24_llama_deepspeed.yaml`

## Testing with MCTS
After training both policy and value networks, we use MCTS (Monte Carlo Tree Search) to solve Game24 problems.

### Step 5: Inference with MCTS
1. First, set the environment variable for MCTS testing:
```bash
export TEST_NO_TERMINAL=1  # Use MCTS for Game24
```

2. Run the test script:
```bash
cd train_mcts_scripts/game24/

# Run test with both policy (CT2) and value models
sh test_policy_and_value.sh \
    ct2_models/llama2_sft_ep3_ct2 \
    checkpoints/game24_scaling_llama2_7b_critic/checkpoint-3
```

The test uses both:
- Policy network (CT2-converted SFT model)
- Value network (critic model)

The test parameters are in `tsllm/offline_rl/test_sft_and_v.py::SearchArgs`:
- Number of simulations
- Search depth
- Exploration settings

