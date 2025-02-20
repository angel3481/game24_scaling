# DeepSpeed Configuration Explanation

## Batch Size and Gradient Settings
- `train_micro_batch_size_per_gpu`: "auto" - Automatically determine optimal batch size per GPU
- `gradient_accumulation_steps`: 2 - Accumulate gradients over 2 steps before updating

## Precision Settings
### BFloat16 (BF16)
- `bf16.enabled`: true - Use BFloat16 precision for better stability

### Float16 (FP16)
- `fp16.enabled`: false - Don't use FP16 precision
- `fp16.min_loss_scale`: 0.0001 - Minimum scale for loss to prevent underflow
- `fp16.fp16_scale_tolerance`: 0.0 - Tolerance for FP16 loss scaling
- `fp16.opt_level`: "O1" - Optimization level for mixed precision

## ZeRO Optimization
- `zero_optimization.stage`: 2 - Use ZeRO stage 2 (partitions optimizer states)
- `zero_optimization.allgather_partitions`: true - Gather partitioned parameters for forward/backward
- `zero_optimization.allgather_bucket_size`: 5e8 - Size of buckets for all-gather operations
- `zero_optimization.contiguous_gradients`: true - Reduce memory fragmentation 