# MemoryforLLM

## Installation

```
# Install Instructions
conda create -n rlllm python=3.8
conda activate rlllm
```


## Main parameters
Parameters can be found in src/arguments.py
```
write_type: write method in memory
memory_k: number of neighbors used at inference time
```

## Train MemoryforLLM benchmarks
```
torchrun --nproc_per_node 1 main.py --normalize_obs --use_attention --max_num_edits 1 --reward_type prob --ac_representation mean --example_sample_type narrowed_example_selection --num-env-steps 5000 --num_actors 1 --num_processes 16 --num-mini-batch 16 --dataset math_qa --write_type max --memory_k 4 --type_embd llama --eval_data test --seed 3 --baseline memory --eval-interval 10 --use_verbalizer
```



