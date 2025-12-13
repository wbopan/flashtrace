# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashTrace is an attribution and interpretability framework for LLMs implementing multiple attribution methods:
- **IFR** (Information Flow Routes) - hook-based residual stream analysis
- **AttnLRP** (Attention-aware Layer-wise Relevance Propagation) - ICML 2024 gradient-based method
- Baseline methods: Integrated Gradients, attention-based, perturbation-based

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run faithfulness evaluation
uv run python evaluations/faithfulness.py --model llama-3B --cuda_num 0 --num_examples 10 --attr_func IG --dataset facts

# Run coverage evaluation
uv run python evaluations/attribution_coverage.py --model llama-3B --cuda 0 --num_examples 10 --attr_func IG --dataset math

# Available models: llama-1B, llama-3B, llama-8B, qwen-1.7B, qwen-4B, qwen-8B
# Available attr_func: IG, attention_I_G, perturbation_all, perturbation_CLP, perturbation_REAGENT, ifr_all_positions, basic, attnlrp, attnlrp_aggregated, attnlrp_aggregated_multi_hop
# Available datasets: math, facts, morehopqa
```

## Architecture

### Core Modules

**ifr_core.py** - IFR computation engine
- `extract_model_metadata()` extracts Llama/Qwen model structure
- `attach_hooks()` captures residual streams via forward hooks
- `compute_ifr_for_all_positions()` computes attribution across all generation tokens
- `compute_multi_hop_ifr()` handles multi-hop reasoning attribution

**llm_attr.py** - Main attribution API
- `LLMAttribution` base class handles tokenization, generation, and prompt tracking
- `LLMIFRAttribution` wraps IFR methods
- `LLMLRPAttribution` wraps AttnLRP methods
- `LLMGradientAttribution` implements Integrated Gradients
- `LLMAttributionResult` contains attribution matrices with sentence-level aggregation via `compute_sentence_attr()`

**lrp_patches.py** - AttnLRP implementation
- `lrp_context()` context manager patches model forward passes with LRP rules
- `detect_model_type()` auto-detects Llama/Qwen architecture
- Custom forward functions apply identity/uniform LRP rules

**lrp_rules.py** - Core LRP autograd functions
- `stop_gradient()` - Identity Rule (Eq. 9)
- `divide_gradient()` - Uniform Rule (Eq. 7)

**shared_utils.py** - Common utilities
- `DEFAULT_PROMPT_TEMPLATE` and `DEFAULT_GENERATE_KWARGS`
- `create_sentences()` - spaCy-based sentence splitting
- `create_sentence_masks()` - binary masks for sentence-token mapping

### Data Flow

```
LLMAttribution.response() → tokenize + generate
    ↓
[IFR] attach_hooks() → capture residuals → compute_ifr_*()
[LRP] lrp_context() → patched forward → backward with LRP rules
    ↓
LLMAttributionResult → token/sentence attribution matrices
    ↓
LLMAttributionEvaluator → faithfulness/coverage metrics
```

## Key Patterns

- **Hook-based capture**: Forward hooks store `pre_attn_resid`, `mid_resid`, `post_resid`, `mlp_out`
- **Chunked processing**: `chunk_tokens` parameter enables memory-efficient long-context handling
- **Chat template abstraction**: Uses tokenizer's `apply_chat_template()`, tracks user prompt indices separately
- **Mixed precision**: Careful dtype management between float16 model weights and float32 attribution computation

## Evaluation Datasets

Located in `data/` directory:
- `math_mine.json` - math reasoning problems
- `10000_facts_9_choose_3.json` - fact-based QA
- `with_human_verification.json` - MoreHopQA dataset

Results saved to `test_results/faithfulness/` and `test_results/attribution_coverage/`.
