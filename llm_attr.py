import torch
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import matplotlib
import matplotlib.cm as mpl_cm
from matplotlib import pyplot as plt
import numpy as np

if not hasattr(mpl_cm, "register_cmap"):
    from matplotlib import colors as _mpl_colors

    def _register_cmap(name=None, cmap=None, data=None, lut=None, *, force=False):
        """Compatibility wrapper for Matplotlib >=3.10 where register_cmap moved."""
        if cmap is not None and data is not None:
            raise ValueError("Cannot specify both `cmap` and `data` when registering a colormap.")
        if data is not None:
            if name is None:
                raise ValueError("Must supply a name when registering colormap data.")
            cmap = _mpl_colors.LinearSegmentedColormap(name, data, lut=lut)
        elif cmap is None:
            raise ValueError("Must supply `cmap` or `data` when registering a colormap.")

        if isinstance(cmap, str):
            cmap = mpl_cm.get_cmap(cmap)

        name = name or cmap.name
        copied = cmap.copy()
        copied.name = name
        mpl_cm._colormaps.register(copied, name=name, force=force)

    def _unregister_cmap(name):
        mpl_cm._colormaps.unregister(name)

    mpl_cm.register_cmap = _register_cmap
    mpl_cm.unregister_cmap = _unregister_cmap

import seaborn as sns
import torch.nn.functional as F
import re
from tqdm import tqdm
from typing import Dict, Any, Optional, Literal
import textwrap
from transformers import LongformerTokenizer, LongformerForMaskedLM
import networkx as nx
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from wordfreq import zipf_frequency

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

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
def newline_cap_split(doc: Doc) -> Doc:
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
def create_sentences(text, tokenizer, return_indices = False, show = False) -> list[str]:
    sentences = []
    separators = []
    indices = []

    # Process the text with spacy
    doc = nlp(text)

    # Extract sentences
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)

    # extract separators
    cur_start = 0
    for sentence in sentences:
        indices.append(cur_start)
        cur_end = text.find(sentence, cur_start)
        separator = text[cur_start:cur_end]
        separators.append(separator)
        cur_start = cur_end + len(sentence)
        # print(repr(separator), repr(sentence))

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
        indices.append(len("".join(sentences)))
        sentences.append(eos)

    indices.append(len(text))

    if return_indices:
        return sentences, indices
    else:
        return sentences

# given words as tokens, and the sentences formed by these tokens,
# create a binary mask of shape [sentences, tokens] where each 
# row has a 1 where a token is in the represented sentence
def create_sentence_masks(tokens, sentences, show = False) -> torch.Tensor:
    # Initialize mask
    mask = torch.zeros((len(sentences), len(tokens)))

    sentence_idx = 0
    sent_pointer = 0  # Pointer in the current sentence

    for token_idx, token in enumerate(tokens):
        current_sentence = sentences[sentence_idx]

        # Assign token to current sentence
        mask[sentence_idx, token_idx] = 1
        
        if '\n' in token:
            sent_pointer += len(token) + 1
        else:
            sent_pointer += len(token)

        # If end of current sentence, move to next
        if sent_pointer >= len(current_sentence):
            sentence_idx += 1
            sent_pointer = 0

        if sentence_idx >= len(sentences):
            break

    return mask

class LLMAttribution():
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

        self.prompt = None
        self.prompt_ids = None
        self.prompt_tokens = None
        self.chat_prompt_indices = None

        self.user_prompt = None
        self.user_prompt_ids = None
        self.user_prompt_tokens = None
        self.user_prompt_indices = None

        self.generation = None
        self.generation_ids = None
        self.generation_tokens = None

        self.model.eval()
    
    def decode_text_into_tokens(self, text) -> list[str]:
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

        ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        text_tokens = []
        offsets = list(offsets)
        for i in range(len(ids)):
            span = offsets.pop(0)
            start, end = span
            actual_text = text[start:end]
            text_tokens.append(actual_text)

        return text_tokens

    def extract_user_prompt_attributions(self, input, attribution) -> list[str]:   
        # Extract all attributions to be kept (gen -> user prompt and gen -> gen attributions)
        user_prompt_attr_idx = torch.tensor(self.user_prompt_indices)
        gen_attr_idx = torch.arange(len(input), attribution.shape[1])
        all_keep_idx = torch.cat((user_prompt_attr_idx, gen_attr_idx), dim = 0)

        return attribution[:, all_keep_idx]
    
    # Takes a torch tensor of size [N, M] and extends it to [N, target_length] with a padding value
    def pad_vector(self, vector, target_length, padding_value = 0) -> torch.Tensor:
        current_length = vector.shape[1]
        if current_length >= target_length:
            return vector
        padding_size = target_length - current_length
        padded_vector = F.pad(vector, (0, padding_size), value=padding_value)
        return padded_vector
    
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
    # This internally saves the input and generated token ids for attribution target
    def response(self, prompt) -> str:
        self.user_prompt = " " + prompt
        self.prompt = self.format_prompt(self.user_prompt)

        # these are the ids for the user supplied prompt
        self.user_prompt_ids = self.tokenizer(self.user_prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids
        # this is the tokenization of the chat prompt
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids

        with torch.no_grad():
            outputs = self.model.generate(self.prompt_ids, **self.generate_kwargs) # [1, num_prompt_tokens + num_generations]

        # Get only the generated tokens (excluding the prompt)
        self.generation_ids = outputs[:, self.prompt_ids.shape[1]:] # [1, num_generations]
        self.generation = self.tokenizer.decode(self.generation_ids[0], skip_special_tokens = True)
        gen_with_eos = self.tokenizer.decode(self.generation_ids[0], skip_special_tokens = False, clean_up_tokenization_spaces = False)

        # we want to find the indices of the formatted prompt that the user prompt occupies
        # we only want to attribute the user prompt, so we track this for later
        n, m = len(self.user_prompt_ids[0]), len(self.prompt_ids[0])
        for i, input_id in enumerate(self.prompt_ids[0]):
            if input_id == self.user_prompt_ids[0, 0]:
                self.user_prompt_indices = list(range(i, i + n)) 
                break

        # make a list of indices which are all prompt tokens 
        # (chat prompt formatting) that are not the user prompt tokens
        self.chat_prompt_indices = [idx for idx in range(0, m) if idx < self.user_prompt_indices[0] or idx > self.user_prompt_indices[-1]]

        # get the full prompt, user prompt, and generation as tokenized words
        self.prompt_tokens = self.decode_text_into_tokens(self.prompt)
        # print(self.prompt_tokens)
        self.user_prompt_tokens = self.decode_text_into_tokens(self.user_prompt)
        # print(self.user_prompt_tokens)
        self.generation_tokens = self.decode_text_into_tokens(gen_with_eos)
        # print(self.generation_tokens)
    
        return self.generation
    
    # nearly identical to response(), but we do not actually query the model
    # we assume generation = target, and generate all the class variables as done in response()
    def target_response(self, prompt, target) -> str:
        self.user_prompt = " " + prompt
        self.prompt = self.format_prompt(self.user_prompt)

        # these are the ids for the user supplied prompt
        self.user_prompt_ids = self.tokenizer(self.user_prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids
        # this is the tokenization of the chat prompt
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids # [1, num_prompt_tokens]
        # Tokenize the target generation
        self.generation_ids = self.tokenizer(target + self.tokenizer.eos_token, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids # [1, num_generations]
        self.generation = target
        gen_with_eos = self.tokenizer.decode(self.generation_ids[0], skip_special_tokens = False, clean_up_tokenization_spaces = False)

        # we want to find which indices of the formatted prompt that the user prompt occupies
        # we will only want to attribute the user prompt, so we track this for later
        n, m = len(self.user_prompt_ids[0]), len(self.prompt_ids[0])
        for i, input_id in enumerate(self.prompt_ids[0]):
            if input_id == self.user_prompt_ids[0, 0]:
                self.user_prompt_indices = list(range(i, i + n)) 
                break

        # make a list of indices which are all prompt tokens 
        # (chat prompt formatting) that are not the user prompt tokens
        self.chat_prompt_indices = [idx for idx in range(0, m) if idx < self.user_prompt_indices[0] or idx > self.user_prompt_indices[-1]]

        # get the full prompt, user prompt, and generation as tokenized words
        self.prompt_tokens = self.decode_text_into_tokens(self.prompt)
        self.user_prompt_tokens = self.decode_text_into_tokens(self.user_prompt)
        self.generation_tokens = self.decode_text_into_tokens(gen_with_eos)

        return self.generation

class LLMAttributionResult():
    def __init__(
        self,
        tokenizer: Any, 
        attribution_matrix: torch.Tensor, 
        prompt_tokens: list[str],
        generation_tokens: list[str],
        all_tokens: Optional[list[str]] = None,
    ) -> None:

        self.tokenizer = tokenizer

        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.all_tokens = all_tokens
        if self.all_tokens is not None:
            self.all_tokens = self.all_tokens

        self.attribution_matrix = attribution_matrix.detach().cpu()

        self.prompt_sentences = None
        self.generation_sentences = None
        self.all_sentences = None

        self.sentence_attr = None
        self.CAGE_sentence_attr = None

    # normalize rows of a matrix to sum to 1
    def normalize_sum_to_one(self, attriubtion_matrix) -> torch.Tensor:
        # we use nans for visualization, but they must be removed (set to 0) for this function
        attribution_no_nan = torch.nan_to_num(attriubtion_matrix, nan=0.0)
        # we do not want to include negative attributions, clamp all to 0
        attribution_no_nan = attribution_no_nan.clamp(0)
        # first, normalize the rows of the attribution matrix to sum to one
        attribution_row_sums = attribution_no_nan.sum(1, keepdim=True) + 1e-8
        # perform normalization
        return attribution_no_nan / attribution_row_sums
    
    def remove_nan(self, attriubtion_matrix) -> torch.Tensor:
        # we use nans for visualization, but they must be removed (set to 0) for this function
        attribution_no_nan = torch.nan_to_num(attriubtion_matrix, nan=0.0)
        # we do not want to include negative attributions, clamp all to 0
        attribution_no_nan = attribution_no_nan.clamp(0)
        return attribution_no_nan

    # normalize the max of a vector to 1
    def normalize_max(self, attribution_vector) -> torch.Tensor:
        if attribution_vector.max() > 0:
            attribution_vector = attribution_vector / attribution_vector.max()
        elif attribution_vector.max() <= 0:
            attribution_vector = - attribution_vector / attribution_vector.min()
        
        return attribution_vector

    ########################################## sentence attr ##########################################

    # This converts any token attribution to a sentence attribution
    def compute_sentence_attr(self, norm = True) -> None:
        # create the prompt ang generation sentences
        self.prompt_sentences = create_sentences("".join(self.prompt_tokens), self.tokenizer)
        self.generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        self.all_sentences = self.prompt_sentences + self.generation_sentences

        # create a mask that tracks the tokens used in each sentence of the generation
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, self.generation_sentences)
        # create a mask that tracks the tokens used in each sentence of the prompt and the generation
        sentence_masks_all = create_sentence_masks(self.prompt_tokens + self.generation_tokens, self.all_sentences)

        num_inp_sent = len(self.prompt_sentences)
        num_gen_sent = len(self.generation_sentences)
        num_all_sent = len(self.all_sentences)

        # Now we want to turn our attribution which is over tokens into an attribution over sentences
        # attribution rows = gen sentences
        # attribution columns = prompt sentences + gen sentences
        self.sentence_attr = torch.full((num_gen_sent, num_all_sent), torch.nan)
        for i in range(num_gen_sent):
            # Select the rows (sentence) of the matrix which are attributed to the inputs (cols)
            # A whole sentence is selected at once
            row_indices = torch.where(sentence_masks_generation[i] == 1)[0]
            attr_rows = self.attribution_matrix[row_indices, :]

            for j in range(num_all_sent):
                # we do not attribute a generation to itself or any
                # future generations so we can skip those here 
                if j > i + num_inp_sent - 1:
                    continue

                # now we select the columns
                col_indices = torch.where(sentence_masks_all[j] == 1)[0]

                # now select a whole sentence of cols from these rows
                attr = attr_rows[:, col_indices]

                # Find which of these indices are NaN
                nan_mask = torch.isnan(attr)
                # Replace NaNs with 0
                attr[nan_mask] = 0.0

                # take sum of this 2d attr and place it in the correct 
                # spot of the sentence attribution
                self.sentence_attr[i, j] = torch.sum(attr)
        
        if norm:
            self.sentence_attr = self.normalize_sum_to_one(self.sentence_attr)
        else:
            self.sentence_attr = self.remove_nan(self.sentence_attr)

        return

    def plot_attr_table_sentence(self, height = None) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return
        
        width = 15
        wrapped_sentences_x = []
        for sentence in self.all_sentences:
            wrapped_sentences_x.append(textwrap.fill(sentence, width=width))
        wrapped_sentences_y = []
        max_num_lines = 0
        for sentence in self.generation_sentences:
            sentence = textwrap.fill(sentence, width=width)
            num_lines = len(sentence.split("\n"))
            max_num_lines = num_lines if num_lines > max_num_lines else max_num_lines
            wrapped_sentences_y.append(sentence)


        fig_width = (len(self.all_sentences) * width / 10) if (len(self.all_sentences) * width / 10) > 10 else 10
        if height is None:
            fig_height = (len(self.generation_sentences) * max_num_lines / 8) if (len(self.generation_sentences) * max_num_lines / 8) > 8 else 8
        else:
            fig_height = 5
            
        fig, axs = plt.subplots(1, 1, figsize = (fig_width, fig_height))

        # use a positive only heatmap cmap
        if np.nanmin(self.sentence_attr) >= 0:
            sns.heatmap(self.sentence_attr, annot=False, xticklabels=wrapped_sentences_x, yticklabels=wrapped_sentences_y, cmap="Blues", ax = axs)
        # use a postitive and negative heatmap cmap
        else:
            # set vmax vmin such that 0 is center value of color map
            max_abs_attr_val = np.nanmax(self.sentence_attr.abs())
            sns.heatmap(self.sentence_attr, annot=False, xticklabels=wrapped_sentences_x, yticklabels=wrapped_sentences_y, vmax=max_abs_attr_val, vmin=-max_abs_attr_val, cmap="Blues", ax = axs)

        axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=200 / len(self.all_sentences))
        plt.yticks(rotation=45) 
        plt.xticks(rotation=45) 
        plt.show()

    def plot_context_attr_sentence(self, title) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return
        
        wrapped_sentences = []
        width = 20
        for sentence in self.prompt_sentences:
            wrapped_sentences.append(textwrap.fill(sentence, width=width))

        fig_width = len(wrapped_sentences) * (width / 10) 
        fig_height = len(wrapped_sentences) / 2 if len(wrapped_sentences) / 2 > 3 else 3

        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(np.arange(len(wrapped_sentences)), self.normalize_max(torch.nansum(self.sentence_attr[:, :len(self.prompt_sentences)].cpu().detach(), dim = 0)))
        plt.xticks(range(len(wrapped_sentences)), wrapped_sentences, rotation=0)
        plt.ylabel("Influence")
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def save_context_attr_sentence(self, prompt_sentences, path) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return
        
        wrapped_sentences = []
        width = 20
        for sentence in prompt_sentences:
            wrapped_sentences.append(textwrap.fill(sentence, width=width))

        fig_width = len(wrapped_sentences) * (width / 10) 
        fig_height = len(wrapped_sentences) / 2 if len(wrapped_sentences) / 2 > 3 else 3

        fig, axs = plt.subplots(1, 1, figsize = (fig_width, fig_height))
        plt.bar(np.arange(len(wrapped_sentences)), self.normalize_max(torch.nansum(self.sentence_attr[:, :len(prompt_sentences)].cpu().detach(), dim = 0)))
        plt.xticks(range(len(wrapped_sentences)), wrapped_sentences, rotation=0)
        plt.ylabel("Influence")
        plt.tight_layout()
        plt.savefig(path + ".png", bbox_inches='tight', transparent = "False")
        fig.clear()
        plt.close(fig)


    def draw_graph(self, cmap = plt.cm.Blues, wrap_width=20, thresh = 0.10, spacing = 4, arrow_mod = 1, rad = 0.3):
        """
        Simplified one-row attribution graph:
        - All tokens (prompts + generations) drawn in one horizontal row
        - Directed weighted edges: generation -> input
        """

        grad_array = self.sentence_attr
        outputs = self.all_sentences
        generated = self.generation_sentences

        grad_array = grad_array.permute((1, 0))  # -> [outputs, generated]
        attr_np = grad_array.cpu().numpy() if hasattr(grad_array, "cpu") else grad_array
        attr_np = np.nan_to_num(attr_np, nan=0.0)

        G = nx.DiGraph()
        prompt_len = len(outputs) - len(generated)
        n_gen = len(generated)

        # Node ids
        prompt_ids = [f"p_{i}" for i in range(prompt_len)]
        gen_ids = [f"g_{j}" for j in range(n_gen)]

        # Add nodes
        def add_node(node_id, label, ntype):
            wrapped = textwrap.fill(label, width=wrap_width)
            wrap_height = len(wrapped.split('\n'))
            G.add_node(node_id, label=wrapped, type=ntype)
            return wrap_height

        max_wrap_height = 0
        for i in range(prompt_len):
            wrap_height = add_node(prompt_ids[i], outputs[i], "prompt")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height
        for j in range(n_gen):
            wrap_height = add_node(gen_ids[j], generated[j], "generated")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height

        def out_i_to_node(i):
            return prompt_ids[i] if i < prompt_len else gen_ids[i - prompt_len]

        # Add edges gen -> output
        for j in range(n_gen):
            src = gen_ids[j]
            for i in range(len(outputs)):
                w = attr_np[i, j] if (i < attr_np.shape[0] and j < attr_np.shape[1]) else 0.0
                if w != 0.0:
                    G.add_edge(src, out_i_to_node(i), weight=w)


        # --- layout: single row ---
        y_row = 0.0
        pos = {}
        all_nodes = prompt_ids + gen_ids
        for idx, nid in enumerate(all_nodes):
            pos[nid] = (idx * spacing, y_row)

        # --- figure ---
        ncols = len(all_nodes)
        fig_width = max(10, ncols * (spacing * 0.6))
        fig, ax = plt.subplots(figsize=(fig_width, 4), dpi = 300)

        # prune edges
        edges = list(G.edges(data=True))
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size > 0:
            threshold = thresh * np.max(np.abs(weights))  # keep edges ≥ 5% of max weight
            for (u, v, edata) in list(edges):  # iterate over a copy
                if abs(edata["weight"]) < threshold:
                    G.remove_edge(u, v)

        # visualization
        edges = G.edges(data=True)  # refresh edges after pruning
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size == 0:
            weights = np.array([1])  # fallback if everything pruned
        max_w = np.max(np.abs(weights))
        norm = mpl.colors.TwoSlopeNorm(vmin=-max_w, vcenter=0, vmax=max_w) \
            if np.min(weights) < 0 else mpl.colors.Normalize(vmin=0, vmax=max_w)
        
        # Draw nodes (larger font + padding)
        for nid, (x, y) in pos.items():
            lbl = G.nodes[nid]["label"]
            ntype = G.nodes[nid]["type"]
            box_color = "#d4c1ffc8" if ntype == "prompt" else "#cfffcc" #cfe8ff
            ax.annotate(
                lbl, xy=(x, y), xytext=(x, y),
                ha="center", va="center", fontsize=12, zorder=3,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color,
                        edgecolor="gray", linewidth=1.2, alpha=1),
            )

        box_height = max_wrap_height / 4
        # Draw edges with curved arrows
        for (u, v, edata) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            start = (x1, y1 - box_height)
            end   = (x2, y2 - box_height)

            w = edata["weight"]
            color = cmap(norm(w))
            width = (1.5 * arrow_mod) + 5.0 * (abs(w) / max_w)

            arrow_rad = rad if x1 <= x2 else -rad
            arrow = FancyArrowPatch(
                (start), (end),
                connectionstyle=f"arc3,rad={arrow_rad}",
                # arrowstyle=f"-|>,head_length={2*arrow_mod},head_width={arrow_mod}",
                arrowstyle=f"<|-,head_length={2*arrow_mod},head_width={arrow_mod}",
                linewidth=width, color=color, alpha=1, zorder=2,
                shrinkA=16, shrinkB=16, mutation_scale=12,
                clip_on=False
            )
            ax.add_patch(arrow)

        ax.set_xlim(-spacing, (ncols - 1) * spacing + spacing)
        ax.set_ylim(-3, 3)
        ax.axis("off")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)


    def save_graph(self, all_sentences, generation_sentences, path, cmap = plt.cm.Blues, wrap_width=20, thresh = 0.10, spacing = 4, arrow_mod = 1, rad = 0.3):
        """
        Simplified one-row attribution graph:
        - All tokens (prompts + generations) drawn in one horizontal row
        - Directed weighted edges: generation -> input
        """

        grad_array = self.sentence_attr
        outputs = all_sentences
        generated = generation_sentences

        grad_array = grad_array.permute((1, 0))  # -> [outputs, generated]
        attr_np = grad_array.cpu().numpy() if hasattr(grad_array, "cpu") else grad_array
        attr_np = np.nan_to_num(attr_np, nan=0.0)

        G = nx.DiGraph()
        prompt_len = len(outputs) - len(generated)
        n_gen = len(generated)

        # Node ids
        prompt_ids = [f"p_{i}" for i in range(prompt_len)]
        gen_ids = [f"g_{j}" for j in range(n_gen)]

        # Add nodes
        def add_node(node_id, label, ntype):
            wrapped = textwrap.fill(label, width=wrap_width)
            wrap_height = len(wrapped.split('\n'))
            G.add_node(node_id, label=wrapped, type=ntype)
            return wrap_height

        max_wrap_height = 0
        for i in range(prompt_len):
            wrap_height = add_node(prompt_ids[i], outputs[i], "prompt")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height
        for j in range(n_gen):
            wrap_height = add_node(gen_ids[j], generated[j], "generated")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height

        def out_i_to_node(i):
            return prompt_ids[i] if i < prompt_len else gen_ids[i - prompt_len]

        # Add edges gen -> output
        for j in range(n_gen):
            src = gen_ids[j]
            for i in range(len(outputs)):
                w = attr_np[i, j] if (i < attr_np.shape[0] and j < attr_np.shape[1]) else 0.0
                if w != 0.0:
                    G.add_edge(src, out_i_to_node(i), weight=w)


        # --- layout: single row ---
        y_row = 0.0
        pos = {}
        all_nodes = prompt_ids + gen_ids
        for idx, nid in enumerate(all_nodes):
            pos[nid] = (idx * spacing, y_row)

        # --- figure ---
        ncols = len(all_nodes)
        fig_width = max(10, ncols * (spacing * 0.6))
        fig, ax = plt.subplots(figsize=(fig_width, 4), dpi = 300)

        # prune edges
        edges = list(G.edges(data=True))
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size > 0:
            threshold = thresh * np.max(np.abs(weights))  # keep edges ≥ 5% of max weight
            for (u, v, edata) in list(edges):  # iterate over a copy
                if abs(edata["weight"]) < threshold:
                    G.remove_edge(u, v)

        # visualization
        edges = G.edges(data=True)  # refresh edges after pruning
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size == 0:
            weights = np.array([1])  # fallback if everything pruned
        max_w = np.max(np.abs(weights))
        norm = mpl.colors.TwoSlopeNorm(vmin=-max_w, vcenter=0, vmax=max_w) \
            if np.min(weights) < 0 else mpl.colors.Normalize(vmin=0, vmax=max_w)
        
        # Draw nodes (larger font + padding)
        for nid, (x, y) in pos.items():
            lbl = G.nodes[nid]["label"]
            ntype = G.nodes[nid]["type"]
            box_color = "#d4c1ffc8" if ntype == "prompt" else "#cfffcc" #cfe8ff
            ax.annotate(
                lbl, xy=(x, y), xytext=(x, y),
                ha="center", va="center", fontsize=12, zorder=3,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color,
                        edgecolor="gray", linewidth=1.2, alpha=1),
            )

        box_height = max_wrap_height / 4
        # Draw edges with curved arrows
        for (u, v, edata) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            start = (x1, y1 - box_height)
            end   = (x2, y2 - box_height)

            w = edata["weight"]
            color = cmap(norm(w))
            width = (1.5 * arrow_mod) + 5.0 * (abs(w) / max_w)

            arrow_rad = rad if x1 <= x2 else -rad
            arrow = FancyArrowPatch(
                (start), (end),
                connectionstyle=f"arc3,rad={arrow_rad}",
                # arrowstyle=f"-|>,head_length={2*arrow_mod},head_width={arrow_mod}",
                arrowstyle=f"<|-,head_length={2*arrow_mod},head_width={arrow_mod}",
                linewidth=width, color=color, alpha=1, zorder=2,
                shrinkA=16, shrinkB=16, mutation_scale=12,
                clip_on=False
            )
            ax.add_patch(arrow)

        ax.set_xlim(-spacing, (ncols - 1) * spacing + spacing)
        ax.set_ylim(-3, 3)
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(path + ".png", dpi=500, transparent = "False")
        fig.clear()
        plt.close(fig)



    ########################################## recursive sentence attr ##########################################

    # this function is identical to compute_recursive_attr except for var names
    # see that function for details
    def compute_CAGE_sentence_attr(self, sentence_to_explain = -1, clear_values = True) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return

        if self.sentence_attr.shape[1] != len(self.all_sentences):
            raise TypeError(
                """This attribution object is of shape [generations, prompt]. 
                This function only operates on attributions of shape 
                [generations, prompt + generations]"""
            )
        
        self.CAGE_sentence_attr = self.sentence_attr[sentence_to_explain].clone()
        gen_row_indices_to_collapse = list(range(0, len(self.generation_sentences[:sentence_to_explain])))[::-1]
        prompt_sentences_length = len(self.prompt_sentences)
        for index in gen_row_indices_to_collapse:
            biased_row = self.sentence_attr[index] * self.CAGE_sentence_attr[prompt_sentences_length + index]
            if clear_values:
                self.CAGE_sentence_attr[prompt_sentences_length + index] = 0
            self.CAGE_sentence_attr += biased_row

        return
        

    ########################################## Multi Sentence Attr ##########################################

    # this function returns a tuple containing a sentence attribution matrix,
    # the sum of all rows of that matrix, the sum of indices_to_explain rows of that matrix, and a CAGE attribution over the indices_to_explain
    def get_all_sentence_attrs(self, indices_to_explain) -> tuple:
        self.compute_sentence_attr(norm = True)

        attr = self.sentence_attr

        row_attr = 0
        for index in indices_to_explain:
            row_attr += attr[index, :]
        row_attr = row_attr.reshape(1, -1)

        rec_attr = 0
        for index in indices_to_explain:
            self.compute_CAGE_sentence_attr(index)
            rec_attr += self.CAGE_sentence_attr
        rec_attr = rec_attr.reshape(1, -1)

        return attr, row_attr, rec_attr
    
class LLMBasicAttribution(LLMAttribution):
    def __init__(self, model, tokenizer, language: str = "en") -> None:
        super().__init__(model, tokenizer)
        self.zipf_language = language

    def calculate_basic_attribution(self, prompt: str, target: Optional[str] = None) -> LLMAttributionResult:
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        prompt_length = len(self.user_prompt_tokens)
        generation_length = len(self.generation_tokens)
        total_length = prompt_length + generation_length

        score_array = torch.zeros((generation_length, total_length), dtype=torch.float32)

        if generation_length == 0:
            all_tokens = self.user_prompt_tokens + self.generation_tokens
            return LLMAttributionResult(
                self.tokenizer,
                score_array,
                self.user_prompt_tokens,
                self.generation_tokens,
                all_tokens=all_tokens,
            )

        if generation_length > 0 and prompt_length > 0:
            normalized_prompt_tokens = [token.strip() for token in self.user_prompt_tokens]

            for gen_idx, gen_token in enumerate(self.generation_tokens):
                normalized_gen_token = gen_token.strip()

                if not normalized_gen_token:
                    continue

                weight = float(zipf_frequency(normalized_gen_token, self.zipf_language))
                if weight <= 0.0:
                    continue

                for prompt_idx, prompt_token in enumerate(normalized_prompt_tokens):
                    if prompt_token == normalized_gen_token:
                        score_array[gen_idx, prompt_idx] = weight

            row_sums = score_array.sum(dim=1, keepdim=True)
            nonzero_rows = row_sums.squeeze(1) > 0
            if torch.any(nonzero_rows):
                score_array[nonzero_rows] = score_array[nonzero_rows] / row_sums[nonzero_rows]

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
        )

class LLMGradientAttribtion(LLMAttribution):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    # if captum version = True, interpolation only performed over prompt tokens
    # else interpolation over prompt tokens and all intermediate generations
    def calculate_IG_per_generation(self, prompt, steps, baseline, batch_size = 1, captum_version = False, target = None) -> LLMAttributionResult:
        # run the model so we can access the input ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
                
        # Make a copy of the input ids
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input, generated, and total tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length

        # instantiate a matrix which will track the attribution of every generated token to the input
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.empty((generation_length, total_length))

        # grads must be measured to the embedding layer
        embedding_layer = self.model.get_input_embeddings()

        # check batch size
        batch_size = steps if steps < batch_size else batch_size

        # create alphas and set step size trapezoidal riemann sum integral estimation
        alphas = torch.linspace(0, 1, steps, dtype=torch.float32).to(self.device)
        step_sizes = torch.full_like((alphas), 1 / steps).to(self.device)
        step_sizes[0] /= 2
        step_sizes[-1] /= 2
        
        # this is used for precision casting in case the model is not loaded in fp32
        model_dtype = next(self.model.parameters()).dtype

        # we calculate the gradients of predicting self.generation_ids[step] 
        # by updating the input to be propmpt + self.generation_ids[:step]
        # for step in tqdm(range(generation_length)):
        for step in range(generation_length):
            # take inputs off of the graph to avoid gradient accumulation across steps
            input_ids_all = input_ids_all.detach()

            # Capture the input embeddings and force require grad
            input_embeds_orig = embedding_layer(input_ids_all).float()
            # The baseline value is a token id. Commonly employed as 0 (for llama that is the token '!')
            # also used is tokenizer.eos_token_id or tokenizer.pad_token_id
            baseline_embeds = embedding_layer(torch.full_like(input_ids_all, baseline)).float()

            # set target as next known generated token
            target_token = self.generation_ids[0, step].item()
   
            # # Make a tensor to store the gradients over all IG steps 
            # # each individual gradient will be [batch_size, seq_len, embedding_dim]
            # IG_grads = torch.zeros((steps, input_embeds_orig.shape[1], input_embeds_orig.shape[2])).to(self.device)

            # Make a tensor to store the sum of the gradients across the IG steps 
            IG_sum = torch.zeros(input_embeds_orig.shape[1], input_embeds_orig.shape[2], device=self.device)

            # perform IG (gradients of interpolated inputs)
            for batch_start in range(0, steps, batch_size):
                # grab a batch of alphas and step sizes
                batch_end = min(batch_start + batch_size, steps)
                alphas_batch = alphas[batch_start : batch_end].view(-1, 1, 1).float()
                step_sizes_batch = step_sizes[batch_start : batch_end].view(-1, 1, 1)

                # interpolate the batch of embeddings
                # captum does not interpolate over the current generated tokens
                # as a result, the generation -> generation gradients are mostly ignored
                if captum_version == True:
                    scaled_embeds_batch = baseline_embeds[:, :input_length] + alphas_batch * (input_embeds_orig[:, :input_length] - baseline_embeds[:, :input_length]) 
                    input_embeds_batch = input_embeds_orig.detach().clone().repeat(batch_end - batch_start, 1, 1)
                    input_embeds_batch[:, :input_length] = scaled_embeds_batch          
                # We do interpolate over the prompt and current generation
                # This allows generation -> generation attributions to be captured
                else:
                    input_embeds_batch = baseline_embeds + alphas_batch * (input_embeds_orig - baseline_embeds) # [batch_size, seq_len, embedding_dim]

                # set requires grad on input embeds
                input_embeds_batch = input_embeds_batch.to(model_dtype).detach().clone().requires_grad_(True)
                # perform inference
                logits = self.model(inputs_embeds=input_embeds_batch).logits # [batch_size, seq_len, vocab_size]
                # evaluate the probability of the target token's generation
                probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1) # [batch_size, vocab_size]
                losses = probs[:, target_token] # [batch_size]

                # clear grads 
                self.model.zero_grad(set_to_none=True)
                if input_embeds_batch.grad is not None:
                    input_embeds_batch.grad.zero_()

                # gather the gradients wrt these probabilities across batch
                losses.sum().backward()

                # perform (x - x') * grad * step_size 
                # baseline_diff = (input_embeds_orig - baseline_embeds) 
                # IG_grads[batch_start : batch_end] = baseline_diff * input_embeds_batch.grad.detach().clone() * step_sizes_batch # [batch_size, seq_len, embedding_dim]

                # perform (x - x') * grad * step_size 
                baseline_diff = (input_embeds_orig - baseline_embeds) 
                grads_batch = baseline_diff * input_embeds_batch.grad.detach().clone() * step_sizes_batch# [batch_size, seq_len, embedding_dim]
                # Sum over batch
                IG_sum += (grads_batch).sum(dim=0) # [seq_len, embedding_dim]

                # Free memory
                del input_embeds_batch, logits, probs, grads_batch
                torch.cuda.empty_cache()

                # del input_embeds_batch, logits, probs, losses

            # # This is a sum over the number of IG steps. Finishes IG result    
            # IG_grads = IG_grads.sum(0) # From [steps, seq_len, embed_dim] to [seq_len, embed_dim]
            # # take the sum over the embedding_dim
            # IG_grads = IG_grads.sum(-1) # [seq_len]

            # Sum across embedding dimension
            IG_grads = IG_sum.sum(-1).detach().cpu() 

            # pad these grads with nan since they must fit into score_array with all other token attributions
            score_array[step] = self.pad_vector(IG_grads.view(1, -1), total_length, torch.nan) # [1, total_length]

            # clean up before the next loop
            # del input_embeds_batch, logits, probs, losses
            # torch.cuda.empty_cache()

            # Append next token to input for next step generation and attribution
            input_ids_all = torch.cat([input_ids_all, torch.tensor([[target_token]]).to(self.device)], dim=1)
            input_ids_all = input_ids_all.detach().clone()

        # remove from the attribution all values associated with thechat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)

class LLMPerturbationAttribution(LLMAttribution):

    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)

        self.mlm_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.mlm_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(self.device)



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


    def compute_kl_response_given_prompt(self, prompt_ids, response_ids) -> torch.Tensor:
        """
        Compute KL divergence scores for each token in `response_ids` given `prompt_ids`.
        Mimics run_probing(metrics="kl_div") but only for the full sequence.

        Args:
            model: HuggingFace autoregressive model.
            prompt_ids: [B, N] tensor of prompt token IDs.
            response_ids: [B, M] tensor of response token IDs.

        Returns:
            KL-divergence scores: [B, M] tensor.
        """
        device = prompt_ids.device
        prompt_ids = prompt_ids.to(device)
        response_ids = response_ids.to(device)

        # Concatenate prompt + response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, N+M]
        attention_mask = torch.ones_like(input_ids, device=device)

        # Compute logits
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, N+M, V]
        logits = logits.to(torch.float32)  # avoid float16 overflow
        log_probs = F.log_softmax(logits, dim=-1)  # [B, N+M, V]

        # Align: y_t predicted from x_{<t}
        B, N = prompt_ids.shape
        M = response_ids.shape[1]
        response_positions = torch.arange(N, N + M, device=device)
        log_probs_response = log_probs[:, response_positions - 1, :]  # [B, M, V]

        # Gather log-probs for actual response tokens
        log_p = log_probs_response.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # [B, M]

        # KL-divergence: assume uniform / null-context q (like run_probing)
        log_p_minus_log_q = -log_probs_response + log_p.unsqueeze(-1)  # [B, M, V]
        p = log_p.exp()  # [B, M]

        kl_scores = (log_p_minus_log_q * p.unsqueeze(-1)).sum(dim=-1)  # [B, M]

        # print(self.tokenizer.decode(response_ids[0]))
        # print(kl_scores)

        return kl_scores


    def calculate_feature_ablation_sentences(self, prompt, baseline, measure = "log_loss", target = None) -> LLMAttributionResult:   
        # run the model so we can access the prompt ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
        
        # Make a copy of the prompt ids 
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input tokens and generated tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length


        # given the text user prompt, create a mask over the tokens of each sentence
        user_prompt_sentences = create_sentences("".join(self.user_prompt_tokens), self.tokenizer, show=True)
        sentence_masks_prompt = create_sentence_masks(self.user_prompt_tokens, user_prompt_sentences, show=True)

        # mask prompt sentences and generated sentences
        # given the generation, create a mask over the tokens of each sentence
        generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, generation_sentences)

        # find the total sizes of the masks we need
        l    = len(self.chat_prompt_indices)     # input formating tokens
        n, m = sentence_masks_prompt.shape       # (user prompt sentences, user prompt tokens)
        o, p = sentence_masks_generation.shape   # (generation sentences + EOS, generation tokens + EOS)

        # Create a tensor that can fit all masks diagonally
        masks = torch.zeros((l + n + o, l + m + p))

        # we never mask the chat_prompt_indices, leave as 0
        # prompt indices cover:
        #   0 : start of sentence_masks_prompt
        #   end of sentence_masks_prompt : start of sentence_masks_generation

        # input sentence masks only cover the user prompt
        user_prompt_start_idx = self.user_prompt_indices[0]
        masks[user_prompt_start_idx : user_prompt_start_idx + n, user_prompt_start_idx : user_prompt_start_idx + m] = sentence_masks_prompt

        # gen sentence masks only cover the generations
        masks[l + n:, l + m:] = sentence_masks_generation
        
        num_input_masks = masks.shape[0]

        # instantiate a matrix which will track the attribution of every generated token to intermediate generations
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.full((generation_length, total_length), torch.nan)

        for step in range(len(sentence_masks_generation)):
        # for step in range(len(sentence_masks_generation) + 1):
            input_ids_all = input_ids_all.detach()

            # assume the we are generating a sentence of the generation_ids and we find the
            # prob of generating this sentence from the current input_ids (prompt + any current generations)
            gen_token_indices = torch.where(sentence_masks_generation[step] == 1)[0] # [len(target_tokens)]
            gen_tokens = self.generation_ids[:, gen_token_indices] # [1, len(target_tokens)]

            if measure == "log_loss":
                original_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]
            elif measure == "KL":
                original_probs = self.compute_kl_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

            # perturb each sentence of the input and current generation 
            # and measure how the probs of predicitng gen_tokens changes
            for i in range(num_input_masks - len(sentence_masks_generation) + step):
                # find the input tokens to be masked
                tokens_to_mask = torch.where(masks[i] == 1)[0]

                # if we don't want to mask anything just continue
                if len(tokens_to_mask) == 0:
                    continue

                # save the original token values for unmasking
                original_token_value = input_ids_all[:, tokens_to_mask].clone()
                # mask the values
                input_ids_all[:, tokens_to_mask] = baseline

                if measure == "log_loss":
                    # prob of generating a token from a perturbation of the input_ids (prompt + current generations)
                    perturbed_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]
                elif measure == "KL":
                    perturbed_probs = self.compute_kl_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

                # change from original generation prob
                score_delta = original_probs - perturbed_probs # [1, len(target_tokens)]

                # since scores are for each output token over the set of input tokens [tokens_to_mask],
                # we expand them to be over all these tokens
                rows, cols = torch.meshgrid(gen_token_indices, tokens_to_mask, indexing = "ij")
                score_array[rows, cols] = score_delta.reshape(-1, 1).repeat((1, len(tokens_to_mask))).to(score_array.dtype) # [len(target_tokens), len(tokens_to_mask)]
                    
                # un-ablate the input
                input_ids_all[:, tokens_to_mask] = original_token_value

            # Append generated tokens to input for next step
            input_ids_all = torch.cat([input_ids_all, gen_tokens], dim = 1)
    
        # remove from the attribution all values associated with the chat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)

    def mlm_mask_indices(self, input_ids, tokens_to_mask):
        """
        Replace masked positions in a LLaMA token sequence using LED.
        """

        # 1. Convert input_ids to tokens
        new_text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 2. Mask only selected tokens
        for idx in tokens_to_mask:
            new_text_tokens[idx] = self.mlm_tokenizer.mask_token

        # 3. Convert tokens back to string
        new_text = self.tokenizer.convert_tokens_to_string(new_text_tokens)

        # 4. Encode for Longformer MLM
        inputs = self.mlm_tokenizer(new_text, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 5. Find masked positions
        masked_positions = (inputs["input_ids"] == self.mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # 6. Add global attention on masked positions
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[0, masked_positions] = 1
        inputs["global_attention_mask"] = global_attention_mask

        # 7. Predict all masked positions in one forward pass
        with torch.no_grad():
            logits = self.mlm_model(**inputs).logits  # [batch, seq_len, vocab_size]
            predicted_ids = logits[0, masked_positions, :].argmax(dim=-1)
            replacement_ids = inputs["input_ids"].clone()
            replacement_ids[0, masked_positions] = predicted_ids

        # 8. Convert predicted tokens to string
        regenerated_tokens = [replacement_ids[0, idx] for idx in masked_positions]
        regenerated_text = self.mlm_tokenizer.decode(predicted_ids, skip_special_tokens=True)
        if regenerated_text and regenerated_text[0] != ' ':
            regenerated_text = ' ' + regenerated_text
            
        # 8. Re-tokenize with LLaMA tokenizer
        replacement_input_ids = self.tokenizer(regenerated_text, return_tensors='pt').input_ids

        # 9. Pad/truncate to match original masked length
        original_len = len(tokens_to_mask)
        new_len = replacement_input_ids.shape[1]

        if new_len > original_len:
            replacement_input_ids = replacement_input_ids[:, :original_len]
        elif new_len < original_len:
            remainder = torch.full((1, original_len - new_len), self.tokenizer.eos_token_id, dtype=torch.long)
            replacement_input_ids = torch.cat((replacement_input_ids, remainder), dim=1)

        if replacement_input_ids.dtype != torch.int64:
            replacement_input_ids = replacement_input_ids.to(torch.int64)

        return replacement_input_ids.to(self.device)

    def calculate_feature_ablation_sentences_mlm(self, prompt, target = None) -> LLMAttributionResult:   
        # run the model so we can access the prompt ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
        
        # Make a copy of the prompt ids 
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input tokens and generated tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length


        # given the text user prompt, create a mask over the tokens of each sentence
        user_prompt_sentences = create_sentences("".join(self.user_prompt_tokens), self.tokenizer, show=True)
        sentence_masks_prompt = create_sentence_masks(self.user_prompt_tokens, user_prompt_sentences, show=True)

        # mask prompt sentences and generated sentences
        # given the generation, create a mask over the tokens of each sentence
        generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, generation_sentences)

        # find the total sizes of the masks we need
        l    = len(self.chat_prompt_indices)     # input formating tokens
        n, m = sentence_masks_prompt.shape       # (user prompt sentences, user prompt tokens)
        o, p = sentence_masks_generation.shape   # (generation sentences + EOS, generation tokens + EOS)

        # Create a tensor that can fit all masks diagonally
        masks = torch.zeros((l + n + o, l + m + p))

        # we never mask the chat_prompt_indices, leave as 0
        # prompt indices cover:
        #   0 : start of sentence_masks_prompt
        #   end of sentence_masks_prompt : start of sentence_masks_generation

        # input sentence masks only cover the user prompt
        user_prompt_start_idx = self.user_prompt_indices[0]
        masks[user_prompt_start_idx : user_prompt_start_idx + n, user_prompt_start_idx : user_prompt_start_idx + m] = sentence_masks_prompt

        # gen sentence masks only cover the generations
        masks[l + n:, l + m:] = sentence_masks_generation
        
        num_input_masks = masks.shape[0]

        # instantiate a matrix which will track the attribution of every generated token to intermediate generations
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.full((generation_length, total_length), torch.nan)

        for step in range(len(sentence_masks_generation)):
        # for step in range(len(sentence_masks_generation) + 1):
            input_ids_all = input_ids_all.detach()

            # assume the we are generating a sentence of the generation_ids and we find the
            # prob of generating this sentence from the current input_ids (prompt + any current generations)
            gen_token_indices = torch.where(sentence_masks_generation[step] == 1)[0] # [len(target_tokens)]
            gen_tokens = self.generation_ids[:, gen_token_indices] # [1, len(target_tokens)]

            original_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

            # perturb each sentence of the input and current generation 
            # and measure how the probs of predicitng gen_tokens changes
            for i in range(num_input_masks - len(sentence_masks_generation) + step):
                # find the input tokens to be masked
                tokens_to_mask = torch.where(masks[i] == 1)[0]

                # if we don't want to mask anything just continue
                if len(tokens_to_mask) == 0:
                    continue

                # save the original token values for unmasking
                # original_token_value = input_ids_all.clone()
                original_token_value = input_ids_all[:, tokens_to_mask].clone()

                # we need replace the tokens_to_mask with roberta predicted words and turn them back into tokens
                new_ids = self.mlm_mask_indices(input_ids_all, tokens_to_mask)
                try:
                    input_ids_all[:, tokens_to_mask] = new_ids
                except:
                    print(new_ids)

                # prob of generating a token from a perturbation of the input_ids (prompt + current generations)
                perturbed_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

                # change from original generation prob
                score_delta = original_probs - perturbed_probs # [1, len(target_tokens)]

                # since scores are for each output token over the set of input tokens [tokens_to_mask],
                # we expand them to be over all these tokens
                rows, cols = torch.meshgrid(gen_token_indices, tokens_to_mask, indexing = "ij")
                score_array[rows, cols] = score_delta.reshape(-1, 1).repeat((1, len(tokens_to_mask))).to(score_array.dtype) # [len(target_tokens), len(tokens_to_mask)]
                    
                # un-ablate the input
                # input_ids_all = original_token_value
                input_ids_all[:, tokens_to_mask] = original_token_value

            # Append generated tokens to input for next step
            input_ids_all = torch.cat([input_ids_all, gen_tokens], dim = 1)
    
        # remove from the attribution all values associated with the chat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)


class LLMAttentionAttribution(LLMAttribution):
    def __init__(self, model, tokenizer, generate_kwargs = None):
        super().__init__(model, tokenizer, generate_kwargs)

    def calculate_attention_attribution(self, prompt, target = None) -> LLMAttributionResult:
        # run the model so we can access the prompt ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
        
        # Make a copy of the input ids 
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input and generated tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length

        # instantiate a matrix which will track the attribution of every generated token to the input
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.empty((generation_length, total_length))

        with torch.no_grad():
            # for step in tqdm(range(generation_length)):
            for step in range(generation_length):
                input_ids_all = input_ids_all.detach()

                target_token = self.generation_ids[0, step]

                # perform inference
                outputs = self.model(input_ids_all, output_attentions = True)
                
                # get attention weights (mean over layers, heads, and attention rows)
                attentions = torch.stack(outputs.attentions, 0).mean(0).mean(1).mean(1) # [batch, seq_length]
                attentions = torch.stack(outputs.attentions, 0)[-1].mean(1).mean(1) # [batch, seq_length]
                # attentions = torch.stack(outputs.attentions, 0)[-1].mean(1).mean(1) # [batch, seq_length]
                # pad the scores with nan since they must fit into score_array with all other token attributions
                score_array[step] = self.pad_vector(attentions.detach().cpu(), total_length, torch.nan)

                # Append generated token to input for next step
                input_ids_all = torch.cat([input_ids_all, torch.tensor([[target_token]]).to(self.device)], dim=1)

        # remove from the attribution all values associated with thechat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)
   
    def rollout(self, attentions):
        num_blocks = attentions.shape[0]
        batch_size = attentions.shape[1]
        num_tokens = attentions.shape[2]
        eye = torch.eye(num_tokens).expand(num_blocks, batch_size, num_tokens, num_tokens).to(attentions[0].device)

        matrices_aug = attentions + eye

        # normalize all the matrices, making residual connection addition equal to 0.5*A + 0.5*I
        matrices_aug = matrices_aug / matrices_aug.sum(dim=-1, keepdim=True)

        # perform rollout
        joint_attention = matrices_aug[0]
        for i in range(0 + 1, num_blocks):
            joint_attention = matrices_aug[i].bmm(joint_attention)

        return joint_attention
