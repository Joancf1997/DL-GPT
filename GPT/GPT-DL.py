



# GPT Configuration to use 
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size             (Tokens)
    "context_length": 1024, # Context length              (Num of words process at the time)
    "emb_dim": 768,         # Embedding dimension         (Embedding)  
    "n_heads": 12,          # Number of attention heads   (Attention)
    "n_layers": 12,         # Number of layers            (MLP)
    "drop_rate": 0.1,       # Dropout rate                (Regularization)
    "qkv_bias": False       # Query-Key-Value bias        (No bias)
}