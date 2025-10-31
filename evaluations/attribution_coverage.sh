

python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 250 --attr_func IG --dataset math
python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 250 --attr_func IG --dataset math
python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 250 --attr_func IG --dataset math
python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 250 --attr_func IG --dataset math

python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math
python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math
python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math
python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 250 --attr_func attention_I_G --dataset math

python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math
python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math
python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math
python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 500 --attr_func perturbation_CLP --dataset math

python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math
python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math
python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math
python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 500 --attr_func perturbation_REAGENT --dataset math

python3 attribution_coverage.py --model llama-3B  --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
python3 attribution_coverage.py --model llama-8B  --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
python3 attribution_coverage.py --model qwen-4B   --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
python3 attribution_coverage.py --model qwen-8B   --cuda_num 0 --num_examples 500 --attr_func perturbation_all --dataset math
