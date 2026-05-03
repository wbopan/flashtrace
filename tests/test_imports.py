def test_public_imports():
    import flashtrace

    assert flashtrace.FlashTrace.__name__ == "FlashTrace"
    assert flashtrace.TraceResult.__name__ == "TraceResult"
    assert callable(flashtrace.load_model_and_tokenizer)
