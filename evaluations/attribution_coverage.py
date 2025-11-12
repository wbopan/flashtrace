import os
import sys
# Ensure project root is importable regardless of CWD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import numpy as np
from transformers import utils
import math
from tqdm import tqdm
import random
import argparse
import csv
from itertools import islice
from typing import Tuple
from huggingface_hub import login

from attribution_datasets import (
    AttributionDataset,
    FactsAttributionDataset,
    MathAttributionDataset,
)

utils.logging.set_verbosity_error()  # Suppress standard warnings

import llm_attr
import llm_attr_eval

utils.logging.set_verbosity_error()

def run_attribution(testing_dict, prompt, batch_size, indices_to_explain = [1], target = None) -> list[torch.Tensor]:
    model = testing_dict["model"]
    tokenizer = testing_dict["tokenizer"]

    # Now we create an attribution for the full response
    if "IG" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMGradientAttribtion(model, tokenizer)

        if testing_dict["attr_func"] == "IG":
            attr = llm_attributor.calculate_IG_per_generation(prompt, 20, tokenizer.eos_token_id, batch_size = batch_size, target = target)

        attributions = attr.get_all_sentence_attrs(indices_to_explain)

    elif "perturbation" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMPerturbationAttribution(model, tokenizer)

        if testing_dict["attr_func"] == "perturbation_all":
            attr = llm_attributor.calculate_feature_ablation_sentences(prompt, baseline = tokenizer.eos_token_id, measure="log_loss", target = target)
        elif testing_dict["attr_func"] == "perturbation_CLP":
            attr = llm_attributor.calculate_feature_ablation_sentences(prompt, baseline = tokenizer.eos_token_id, measure="KL", target = target)
        elif testing_dict["attr_func"] == "perturbation_REAGENT":
            attr = llm_attributor.calculate_feature_ablation_sentences_mlm(prompt, target = target)

        attributions = attr.get_all_sentence_attrs(indices_to_explain)
        
    elif "attention" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMAttentionAttribution(model, tokenizer)
        llm_attributor_ig = llm_attr.LLMGradientAttribtion(model, tokenizer)

        if testing_dict["attr_func"] == "attention_I_G":
            attr = llm_attributor.calculate_attention_attribution(prompt, target = target)
            attr_b = llm_attributor_ig.calculate_IG_per_generation(prompt, 20, tokenizer.eos_token_id, batch_size = batch_size, target = target)
            attr.attribution_matrix = attr.attribution_matrix * attr_b.attribution_matrix

        attributions = attr.get_all_sentence_attrs(indices_to_explain)       

    elif "ifr" in testing_dict["attr_func"].lower():
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        attr_func = testing_dict["attr_func"].lower()
        renorm_threshold = testing_dict.get("renorm_threshold")

        if attr_func == "ifr_all_positions":
            attr = llm_attributor.calculate_ifr_for_all_positions(prompt, target=target, renorm_threshold=renorm_threshold)
        elif attr_func == "ifr_span":
            span = testing_dict.get("sink_span")
            attr = llm_attributor.calculate_ifr_span(
                prompt,
                target=target,
                span=tuple(span) if span is not None else None,
                renorm_threshold=renorm_threshold,
            )
        elif attr_func == "ifr_multi_hop":
            attr = llm_attributor.calculate_ifr_multi_hop(
                prompt,
                target=target,
                sink_span=tuple(testing_dict.get("sink_span")) if testing_dict.get("sink_span") is not None else None,
                thinking_span=tuple(testing_dict.get("thinking_span")) if testing_dict.get("thinking_span") is not None else None,
                n_hops=testing_dict.get("n_hops", 1),
                renorm_threshold=renorm_threshold,
                observation_mask=testing_dict.get("observation_mask"),
            )
        else:
            raise ValueError(f"Unsupported IFR attribution function '{testing_dict['attr_func']}'.")

        # Sentence-level aggregation expected downstream
        attributions = attr.get_all_sentence_attrs(indices_to_explain)

    elif "basic" in testing_dict["attr_func"]:
        llm_attributor = llm_attr.LLMBasicAttribution(model, tokenizer)
        attr = llm_attributor.calculate_basic_attribution(prompt, target = target)
        attributions = attr.get_all_sentence_attrs(indices_to_explain)

    return attributions


def attribution_coverage(testing_dict, llm_evaluator, prompt, attr_mask_indices, indices_to_explain, target = None) -> np.ndarray[float]:
    tokenizer = testing_dict["tokenizer"]

    prompt_sentences = llm_attr_eval.create_sentences(" " + prompt, tokenizer)

    # batch size is set based on the max_input_len in main(). Currently set to fully fill a 196GB GPU.
    if target is None:
        _, full_output = llm_evaluator.response(prompt)
        batch_size = math.floor((testing_dict["max_input_len"] - 100) / len(tokenizer(full_output).input_ids))
    else:
        batch_size = math.floor((testing_dict["max_input_len"] - 100) / len(tokenizer(llm_evaluator.format_prompt(prompt) + target).input_ids))

    attr_list = run_attribution(testing_dict, prompt, batch_size, indices_to_explain = indices_to_explain, target = target)

    scores = []
    for i in range(len(attr_list)):
        if testing_dict["coverage"] in ("prompt", "input"):
            full_attr = attr_list[i][:, :len(prompt_sentences)]
        elif testing_dict ["coverage"] == "all":   
            full_attr = attr_list[i]
        else:
            raise ValueError(f"Unsupported coverage option '{testing_dict['coverage']}'. Expected 'prompt' or 'all'.")

        partial_attr = attr_list[i][:, attr_mask_indices]
        scores.append(llm_evaluator.evaluate_attr_coverage(full_attr, partial_attr))

    return np.array(scores)

def clean_trailing_space(text) -> str:
    if text[-1] == ' ':
        return text[:-1]
    else:
        return text
    
def evaluate_attribution(testing_dict) -> None:
    model = testing_dict["model"]
    tokenizer = testing_dict["tokenizer"]

    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(model, tokenizer)

    scores = []

    description = "Attribution Coverage 2 " + testing_dict["model_name"] + " " + testing_dict["dataset_name"] + " " + testing_dict["attr_func"]

    dataset: AttributionDataset = testing_dict["dataset"]
    num_examples = testing_dict["num_examples"]
    total = min(len(dataset), num_examples) if hasattr(dataset, "__len__") else num_examples
    example_iterator = islice(dataset, num_examples)

    for example in tqdm(example_iterator, desc=description, total=total):
        if example.attr_mask_indices is None:
            continue

        indices_to_explain = example.indices_to_explain if example.indices_to_explain is not None else [-2]
        scores.append(
            attribution_coverage(
                testing_dict,
                llm_evaluator,
                example.prompt,
                example.attr_mask_indices,
                indices_to_explain=indices_to_explain,
                target=example.target,
            )
        ) # [num_attrs, 3 or 4 scores]

    scores = np.array(scores) # [num_examples, num_attrs, 3 scores]
    scores_mean = scores.mean(0) # [num_attrs, 3 scores]
    scores_var = scores.std(0) # [num_attrs, 3 scores]

    # make the test folder if it doesn't exist
    folder = "./test_results/attribution_coverage_2_" + testing_dict["coverage"] + "/" + testing_dict["dataset_name"] + "/" + testing_dict["model_name"] + "/" 
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save all data
    file_name = testing_dict["attr_func"] + "_" + str(testing_dict["num_examples"]) + "_examples"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)

        write.writerow(["Method", "Double Bound", "Lower Bound", "Mean Bound"])

        write.writerow(["Seq Attr Scores Mean"] + scores_mean[0].tolist())
        write.writerow(["Row Attr Scores Mean"] + scores_mean[1].tolist())
        write.writerow(["Recursive Attr Scores Mean"] + scores_mean[2].tolist())

        write.writerow(["Seq Attr Scores Var"] + scores_var[0].tolist())
        write.writerow(["Row Attr Scores Var"] + scores_var[1].tolist())
        write.writerow(["Recursive Attr Scores Var"] + scores_var[2].tolist())

    return

def load_model(model_name, device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

    # Respect three modes:
    # - device == 'auto'               -> multi-GPU sharding across all visible devices
    # - device startswith('cuda:IDX')   -> place entire model on a single GPU IDX (relative to visible devices)
    # - device == 'cpu'                -> CPU
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_idx = int(device.split(":")[1])
        except Exception:
            gpu_idx = 0
        # Map whole model to a single GPU index using an explicit device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_idx},  # put entire model on one GPU
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    else:
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main(args) -> None:
    # login(token = "")

    # Device selection policy:
    # - If --cuda is a comma-separated list (e.g. "0,1"), set visibility to that list and shard with device_map='auto'.
    # - If --cuda is a single index (e.g. "0"), do NOT override CUDA_VISIBLE_DEVICES; place model on cuda:{index}.
    # - Else (no --cuda), use --cuda_num as single-device index relative to current visibility.
    device: str
    if args.cuda is not None and isinstance(args.cuda, str) and "," in args.cuda:
        # Multi-GPU sharding across the provided visible set
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        device = "auto"
    elif args.cuda is not None and isinstance(args.cuda, str) and args.cuda.strip() != "":
        # Single-GPU by relative index; do not modify CUDA_VISIBLE_DEVICES here
        try:
            idx = int(args.cuda)
        except Exception:
            idx = 0
        device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    else:
        # Fallback to cuda_num
        device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # set up model
    if args.model == "llama-1B":
        model_name = "meta-llama/Llama-3.2-1B-Instruct" 
        max_input_len = 5500
    elif args.model == "llama-3B":
        model_name = "meta-llama/Llama-3.2-3B-Instruct" 
        max_input_len = 4800
    elif args.model == "llama-8B":
        model_name = "meta-llama/Llama-3.1-8B-Instruct" 
        max_input_len = 3500
    elif args.model == "qwen-1.7B":
        model_name = "Qwen/Qwen3-1.7B" 
        max_input_len = 5500
    elif args.model == "qwen-4B":
        model_name = "Qwen/Qwen3-4B" 
        max_input_len = 3500
    elif args.model == "qwen-8B":
        model_name = "Qwen/Qwen3-8B" 
        max_input_len = 3000
    elif args.model == "qwen-32B":
        model_name = "Qwen/Qwen3-32B"
        max_input_len = 1500
    elif args.model == "gemma-12B":
        model_name = "gemma/gemma-3-12b-it"
        max_input_len = 1500
    elif args.model == "gemma-27B":
        model_name = "gemma/gemma-3-27b-it"
        max_input_len = 2000
    else:
        model_name = args.model_path if args.model_path is not None else args.model
        max_input_len = 2000

    model, tokenizer = load_model(model_name if args.model_path is None else args.model_path, device)

    dataset_registry = {
        "math": lambda: MathAttributionDataset("./data/math_mine.json", tokenizer),
        "facts": lambda: FactsAttributionDataset("./data/10000_facts_9_choose_3.json"),
    }
    dataset_loader = dataset_registry.get(args.dataset)
    if dataset_loader is None:
        print("You have not specified an acceptable dataset. Exiting.")
        exit()
    dataset = dataset_loader()

    testing_dict = {
        "model" : model,
        "model_name": args.model,
        "tokenizer" : tokenizer,
        "dataset" : dataset,
        "dataset_name" : args.dataset,
        "coverage" : args.coverage,
        "max_input_len": max_input_len,
        "attr_func": args.attr_func,
        "num_examples": args.num_examples,
        "device": device
    }

    # call the test function
    evaluate_attribution(testing_dict)

    return

if __name__ == "__main__":    
    parser = argparse.ArgumentParser('')
    parser.add_argument('--num_examples',
                        type = int, default = 100,
                        help='How many dataset examples to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "llama",
                        help='Model to use: llama or qwen')
    parser.add_argument('--model_path',
                        type=str, default=None,
                        help='Optional local model path to load (overrides model repo id only).')
    parser.add_argument('--attr_func',
                        type = str,
                        default = "IG",
                        help="attr to use: \
                            grad, IG, IG_captum, contextcite, attention, rollout, perturbation \
                        ")
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='Single GPU index to use (e.g., 0).')
    parser.add_argument('--cuda',
                        type=str, default=None,
                        help='GPU selection: use comma-separated ids for multi-GPU sharding (e.g. "0,1"); use a single index for one GPU relative to current CUDA_VISIBLE_DEVICES (e.g. "0").')
    parser.add_argument('--dataset',
            type = str, default = "math",
            help = 'The dataset to evaluate on: math or facts')
    parser.add_argument('--coverage',
            type = str, default = "prompt",
            help = 'The attributions over which to measure coverage. prompt: only prompt. all: prompt and gen.')
    
    args, unparsed = parser.parse_known_args()
    
    main(args)
