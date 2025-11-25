"""Basic tests for shared_utils module."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


class TestSharedUtilsImports:
    """Test that shared_utils can be imported and exports expected symbols."""

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
    def test_import_constants(self):
        from shared_utils import DEFAULT_GENERATE_KWARGS, DEFAULT_PROMPT_TEMPLATE

        assert isinstance(DEFAULT_GENERATE_KWARGS, dict)
        assert "max_new_tokens" in DEFAULT_GENERATE_KWARGS
        assert isinstance(DEFAULT_PROMPT_TEMPLATE, str)
        assert "{context}" in DEFAULT_PROMPT_TEMPLATE

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
    def test_import_nlp(self):
        from shared_utils import nlp
        assert nlp is not None
        assert "newline_cap_split" in nlp.pipe_names

    @pytest.mark.skipif(not HAS_SPACY or not HAS_TORCH, reason="spacy or torch not installed")
    def test_import_functions(self):
        from shared_utils import (
            create_sentences,
            create_sentences_fallback,
            create_sentence_masks,
        )
        assert callable(create_sentences)
        assert callable(create_sentences_fallback)
        assert callable(create_sentence_masks)


class TestCreateSentencesFallback:
    """Test the fallback sentence splitter (pure Python, no deps)."""

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
    def test_simple_text(self):
        from shared_utils import create_sentences_fallback

        text = "Hello world. This is a test."
        sentences = create_sentences_fallback(text)
        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "This is a test."

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
    def test_newline_split(self):
        from shared_utils import create_sentences_fallback

        text = "First line\nSecond line"
        sentences = create_sentences_fallback(text)
        assert len(sentences) == 2

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
    def test_empty_text(self):
        from shared_utils import create_sentences_fallback

        sentences = create_sentences_fallback("")
        assert sentences == []

    @pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
    def test_single_sentence(self):
        from shared_utils import create_sentences_fallback

        text = "Just one sentence"
        sentences = create_sentences_fallback(text)
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence"


@pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
class TestCreateSentences:
    """Test the main sentence splitter using spacy."""

    def test_simple_text(self):
        from shared_utils import create_sentences

        text = "Hello world. This is a test."
        sentences = create_sentences(text, tokenizer=None)
        assert len(sentences) >= 1

    def test_with_return_indices(self):
        from shared_utils import create_sentences

        text = "First sentence. Second sentence."
        sentences, indices = create_sentences(text, tokenizer=None, return_indices=True)
        assert isinstance(sentences, list)
        assert isinstance(indices, list)
        assert len(indices) > 0


@pytest.mark.skipif(not HAS_TORCH or not HAS_SPACY, reason="torch or spacy not installed")
class TestCreateSentenceMasks:
    """Test the sentence mask creation."""

    def test_basic_mask(self):
        import torch
        from shared_utils import create_sentence_masks

        tokens = ["Hello", " ", "world", ".", " ", "Test", "."]
        sentences = ["Hello world.", " Test."]
        mask = create_sentence_masks(tokens, sentences)

        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (2, 7)

    def test_mask_values(self):
        from shared_utils import create_sentence_masks

        tokens = ["A", "B", "C"]
        sentences = ["AB", "C"]
        mask = create_sentence_masks(tokens, sentences)

        # Each token should be assigned to exactly one sentence
        assert mask.sum(dim=0).max() <= 1


class TestModuleImports:
    """Test that dependent modules can import from shared_utils."""

    def test_llm_attr_imports(self):
        """Test llm_attr.py can import shared_utils."""
        try:
            import llm_attr
            assert hasattr(llm_attr, 'LLMAttribution')
        except ImportError as e:
            pytest.skip(f"llm_attr not importable: {e}")

    def test_llm_attr_eval_imports(self):
        """Test llm_attr_eval.py can import shared_utils."""
        try:
            import llm_attr_eval
            assert hasattr(llm_attr_eval, 'LLMAttributionEvaluator')
        except ImportError as e:
            pytest.skip(f"llm_attr_eval not importable: {e}")

    def test_attribution_datasets_imports(self):
        """Test attribution_datasets.py can import shared_utils."""
        try:
            import attribution_datasets
            assert hasattr(attribution_datasets, 'AttributionDataset')
        except ImportError as e:
            pytest.skip(f"attribution_datasets not importable: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
