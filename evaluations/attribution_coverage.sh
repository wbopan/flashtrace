# python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math
# python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math
# python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math
# python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math

# python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math
# python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math
# python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math
# python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math

# python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math
# python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math
# python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math
# python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math

# python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
# python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
# python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
# python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math

# test
# export CUDA_VISIBLE_DEVICES=1,2
# python3 evaluations/attribution_coverage.py --model qwen-8B --model_path /opt/share/models/Qwen/Qwen3-8B/ --cuda '0,1' --num_examples 500 --attr_func ifr_multi_hop --dataset math
# python3 evaluations/attribution_coverage.py --model qwen-32B --model_path /opt/share/models/Qwen/Qwen3-32B/ --cuda '0,1' --num_examples 500 --attr_func ifr_multi_hop --dataset math

CUDA_VISIBLE_DEVICES=6,7 python3 evaluations/attribution_coverage.py --model qwen-8B --model_path /opt/share/models/Qwen/Qwen3-8B/ --cuda '0,1' --num_examples 50 --attr_func ifr_multi_hop --dataset math
