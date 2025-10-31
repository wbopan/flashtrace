from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import numpy as np
from transformers import utils
import math
from tqdm import tqdm
import random
from datasets import load_dataset
import argparse
import os
import csv
from typing import Tuple
import json
from huggingface_hub import login

utils.logging.set_verbosity_error()  # Suppress standard warnings

os.sys.path.append(os.path.dirname(os.path.abspath('.')))

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
        if testing_dict["coverage"] == "prompt":
            full_attr = attr_list[i][:, :len(prompt_sentences)]
        elif testing_dict ["coverage"] == "all":   
            full_attr = attr_list[i]

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

    if testing_dict["dataset_name"] == "math":
        for i, example in enumerate(tqdm(testing_dict["dataset"][0 : testing_dict["num_examples"]], desc = description)): 
            dataset_item = example["question"]
            sentences = llm_attr_eval.create_sentences(dataset_item, tokenizer)
            context_sentences = sentences[:-1]
            question = sentences[-1][1:]

            # add dummy sentences to prompt
            context_sentences_w_dummy = llm_evaluator.add_dummy_facts_to_prompt(context_sentences)
            question_sentences_w_dummy = llm_evaluator.add_dummy_facts_to_prompt([question])
            prompt = "".join(context_sentences_w_dummy) + "\n" + "".join(question_sentences_w_dummy)
            # attribution should only be on odd indices
            attr_mask = torch.zeros((1, len(context_sentences_w_dummy) + len(question_sentences_w_dummy)))
            attr_mask[:, ::2] = 1
            attr_mask_indices = torch.where(attr_mask[0] == 1)[0]

            scores.append(attribution_coverage(testing_dict, llm_evaluator, prompt, attr_mask_indices, indices_to_explain = [-2])) # [num_attrs, 3 scores]

    elif testing_dict["dataset_name"] == "facts":
        for i, example in enumerate(tqdm(testing_dict["dataset"][0 : testing_dict["num_examples"]], desc = description)): 
            prompt = example["prompt"]
            target = example["target"]
            attr_mask_indices = example["attr_mask_indices"]
            indices_to_explain = example["indices_to_explain"]

            scores.append(attribution_coverage(testing_dict, llm_evaluator, prompt, attr_mask_indices, indices_to_explain = indices_to_explain, target = target)) # [num_attrs, 4 scores]

    scores = np.array(scores) # [num_examples, num_attrs, 3 scores]
    scores_mean = scores.mean(0) # [num_attrs, 3 scores]
    scores_var = scores.std(0) # [num_attrs, 3 scores]

    # make the test folder if it doesn't exist
    folder = "../test_results/attribution_coverage_2_" + testing_dict["coverage"] + "/" + testing_dict["dataset_name"] + "/" + testing_dict["model_name"] + "/" 
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device, # dispatch efficiently the model on the available ressources
        attn_implementation="eager",
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main(args) -> None:
    login(token = "")

    device = 'cuda:' + str(args.cuda_num) if torch.cuda.is_available() else 'cpu'

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

    model, tokenizer = load_model(model_name, device)

    # set up dataset
    if args.dataset == "math":
        with open("../data/math_mine.json", "r") as f:
            dataset = json.load(f)
    elif args.dataset == "facts":
        with open("../data/10000_facts_9_choose_3.json", "r") as f:
            dataset = json.load(f)    
    else:
        print("You have not specified an acceptable dataset. Exiting.")
        exit()

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
    parser.add_argument('--attr_func',
                        type = str,
                        default = "IG",
                        help="attr to use: \
                            grad, IG, IG_captum, contextcite, attention, rollout, perturbation \
                        ")
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--dataset',
            type = str, default = "squad",
            help = 'The dataset to evaluate on: squad or hotpotqa')
    parser.add_argument('--coverage',
            type = str, default = "input",
            help = 'The attributions over which to measure coverage. prompt: only prompt. all: prompt and gen.')
    
    args, unparsed = parser.parse_known_args()
    
    main(args)