#!/bin/bash
# Attribution coverage evaluation script

CUDA_VISIBLE_DEVICES=0   python3 evaluations/attribution_coverage.py --model gemma-12B --model_path /opt/share/models/gemma/gemma-3-12b-it  --cuda '0'   --num_examples 250 --attr_func IG --dataset math &
CUDA_VISIBLE_DEVICES=3,4 python3 evaluations/attribution_coverage.py --model gemma-27B --model_path /opt/share/models/gemma/gemma-3-27b-it  --cuda '0,1' --num_examples 250 --attr_func IG --dataset math &
CUDA_VISIBLE_DEVICES=1,2   python3 evaluations/attribution_coverage.py --model qwen-8B   --model_path /opt/share/models/Qwen/Qwen3-8B/       --cuda '0,1'   --num_examples 250 --attr_func IG --dataset math &
CUDA_VISIBLE_DEVICES=5,6,7 python3 evaluations/attribution_coverage.py --model qwen-32B  --model_path /opt/share/models/Qwen/Qwen3-32B/      --cuda '0,1,2' --num_examples 250 --attr_func IG --dataset math &
wait

CUDA_VISIBLE_DEVICES=6,7 python3 evaluations/attribution_coverage.py --model qwen-8B --model_path /opt/share/models/Qwen/Qwen3-8B/ --cuda '0,1' --num_examples 50 --attr_func ifr_multi_hop --dataset math
