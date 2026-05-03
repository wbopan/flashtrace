class FlashTrace:
    """Public facade for FlashTrace attribution."""

    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.options = dict(kwargs)
