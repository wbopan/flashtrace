import torch
import numpy as np
import re
from typing import Dict, Any, Optional, Tuple, List
from evaluate import load
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer, util
import string
import math

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context:{context}\n\n\nQuery: {query}"

# sentence detector
try:
    nlp = spacy.load("en_core_web_sm")
    _newline_pipe_position = {"before": "parser"}
except OSError:
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _newline_pipe_position = {"after": "sentencizer"} if "sentencizer" in nlp.pipe_names else {"last": True}

# Custom component to split on capitalized words after newline
@Language.component("newline_cap_split")
def newline_cap_split(doc) -> Doc:
    for i, token in enumerate(doc):
        if token.is_title and i > 0:
            prev_token = doc[i - 1]
            # Check if there's a newline in the previous token or if next token starts with '='
            if "\n" in prev_token.text or (prev_token.is_space and "\n" in prev_token.text):
                token.is_sent_start = True
    return doc

# Add to pipeline *before* parser
nlp.add_pipe("newline_cap_split", **_newline_pipe_position)

# Split text into sentences and return the sentences
def create_sentences(text, tokenizer) -> list[str]:
    sentences = []
    separators = []

    # Process the text with spacy
    doc = nlp(text)

    # Extract sentences
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)

    # extract separators
    cur_start = 0
    for sentence in sentences:
        cur_end = text.find(sentence, cur_start)
        separator = text[cur_start:cur_end]
        separators.append(separator)
        cur_start = cur_end + len(sentence)

    # combine the separators with the sentences properly
    for i in range(len(sentences)):
        if separators[i] == "\n":
            sentences[i] = sentences[i] + separators[i]
        else:
            sentences[i] = separators[i] + sentences[i]  

    # if the text had an eos token (generated text) it will be missed
    # and attached on the last sentence, so we manually handle it
    eos = tokenizer.eos_token
    if eos in sentences[-1]:
        sentences[-1] = sentences[-1].replace(eos, "")
        sentences.append(eos)

    return sentences

class LLMAttributionEvaluator():
    def __init__(
        self, 
        model: Any, 
        tokenizer: Any, 
        generate_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.generated_ids = None
        self.prompt_ids = None
        
        self.model.eval()

        self.squad_metric = load("squad")
        self.bertscore = load("bertscore")
        self.sentence_sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
    
    def format_prompt(self, prompt) -> str:
        modified_prompt = DEFAULT_PROMPT_TEMPLATE.format(context = prompt, query = "")
        formatted_prompt = [{"role": "user", "content": modified_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            formatted_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        return formatted_prompt

    # Query the model for its generation
    # This internally saves the input and generated token ids
    def response(self, prompt) -> Tuple[str, str]:
        formatted_prompt = self.format_prompt(" " + prompt)

        model_input = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens = False).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(model_input.input_ids, **self.generate_kwargs) # [1, num_prompt_tokens + num_generations]
            # Get only the prompt tokens (excluding the prompt)
            self.prompt_ids = outputs[:, :model_input.input_ids.shape[1]] # [1, num_prompt_tokens]
            # Get only the generated tokens (excluding the prompt)
            self.generated_ids = outputs[:, model_input.input_ids.shape[1]:] # [1, num_generations]

        return self.tokenizer.decode(self.generated_ids[0], skip_special_tokens=True), self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    #  we want to evaluate the probability of producing a reponse given a prompt
    def compute_logprob_response_given_prompt(self, prompt_ids, response_ids) -> torch.Tensor:
        """
        Compute log-probabilities of `response_ids` given `prompt_ids`.

        prompt_ids: [B, N]
        response_ids: [B, M]
        Returns: [B, M]
        """
        # concat prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)   # [B, N+M]
        attention_mask = torch.ones_like(input_ids)

        # Get model outputs
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, seq_len, vocab_size]

        # Compute log-probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]

        # Only consider response tokens
        response_start = prompt_ids.shape[1]

        # Align logits to predict each y_t from y_{<t}
        logits_for_response = log_probs[:, response_start - 1: -1, :]  # [B, M, vocab]

        # Gather log-probs for the actual response tokens
        gathered = logits_for_response.gather(2, response_ids.unsqueeze(-1))  # [B, M, 1]
        return gathered.squeeze(-1)  # [B, M]


    def get_topk_tokens(self, attr_matrix, text_list, topk = 10) -> torch.Tensor:
        input_len = len(text_list)
        input_col_sums = attr_matrix.sum(0).clamp(0)[0 : input_len]
        topk_cols = torch.topk(input_col_sums, topk)[1]

        return torch.sort(topk_cols)[0]

    def add_dummy_facts_to_prompt(self, text_sentences) -> List[str]:
        # create dummy fact sentences
        dummy_sentences = []
        for i in range(len(text_sentences)):
            dummy_sentences.append(" Unrelated Sentence.")

        # Interleave the dummy facts
        result = []
        for x, y in zip(text_sentences, dummy_sentences):
            result.append(x)
            result.append(y)

        # add back on the last sentence that we left out
        return result

    def faithfulness_test(self, attribution, segmented_prompt, generation) -> float:        
        
        def get_score_of_prompt(segmented_prompt):
            prompt = "".join(segmented_prompt)
            if prompt[0] == " ":
                prompt = prompt[1:]
            formatted_prompt = self.format_prompt(prompt)
            prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens = False).input_ids.to(self.device)
            generation_ids = self.tokenizer(generation + self.tokenizer.eos_token, return_tensors="pt", add_special_tokens = False).input_ids.to(self.device)

            return self.compute_logprob_response_given_prompt(prompt_ids, generation_ids).sum().cpu().detach().item()
        
        def auc(arr):
            return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

        segmented_prompt = segmented_prompt.copy()

        scores = np.zeros(len(segmented_prompt) + 1)
        density = np.zeros(len(segmented_prompt) + 1)

        # score the unperturbed model input
        scores[0] = get_score_of_prompt(segmented_prompt)
        density[0] = 1

        # find the ranking of prompt sentences by attr magnitude
        sorted_attr_indices = torch.sort(attribution[:, :len(segmented_prompt)].sum(0), descending=True)[1]
        
        attr_sum = attribution.sum()

        for i, idx in enumerate(sorted_attr_indices):
            # find sentence to perturb and replace it with all eos tokens
            selected_text = segmented_prompt[idx]
            selected_text_tokens = self.tokenizer(selected_text, add_special_tokens = False).input_ids
            segmented_prompt[idx] = self.tokenizer.eos_token * len(selected_text_tokens)
            # captured perturbed score and attribution of perturbation
            scores[i + 1] = get_score_of_prompt(segmented_prompt)
            density[i + 1] = density[i] - (attribution.sum(0)[idx] / attr_sum)

        min_normalized_pred = 1.0
        # perform monotonic normalization of raw model response
        normalized_model_response = scores.copy()
        for i in range(len(scores)):           
            normalized_pred = (normalized_model_response[i] - scores[-1]) / (abs(scores[0] - scores[-1]))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            min_normalized_pred = min(min_normalized_pred, normalized_pred)
            normalized_model_response[i] = min_normalized_pred

        alignment_penalty = np.abs(normalized_model_response - density)
        corrected_scores = normalized_model_response + alignment_penalty
        # scores should be clipped before normalization or else values outside of these bounds will artificially improve the final score
        corrected_scores = corrected_scores.clip(0, 1)
        corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))

        if np.isnan(corrected_scores).any():
            corrected_scores = np.linspace(1, 0, len(scores))

        return auc(normalized_model_response), auc(corrected_scores), auc(normalized_model_response + alignment_penalty)

    def evaluate_attr_coverage(self, full_attr, partial_attr) -> np.ndarray[float]:
        # we want all of the attr magnitude placed across each sentence
        whole = full_attr.sum(0).abs()
        # for the portion, we want postively attributed sentences
        portion = partial_attr.sum(0).abs()

        expectation = 1 / portion.shape[0]
        expectation_low = expectation - expectation / 2
        expectation_high = expectation + expectation / 2

        total_attr = whole.sum()
        if total_attr.item() == 0:
            return 0.0, 0.0, 0.0

        ratios = portion / total_attr

        coverage_rate_a = torch.where((ratios > expectation_low) & (ratios < expectation_high))[0].shape[0] / portion.shape[0]
        coverage_rate_b = torch.where((ratios > expectation_low))[0].shape[0] / portion.shape[0]
        coverage_rate_c = torch.where((ratios > expectation))[0].shape[0] / portion.shape[0]

        return coverage_rate_a, coverage_rate_b, coverage_rate_c
    
