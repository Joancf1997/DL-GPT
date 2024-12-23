import torch 
import torch.nn as nn 


"""
  MultiHead Attention
    d_in -> (Embedding dimension) Input dim to the transformer block 
    d_out -> Output dimension of the transformer block (usually the same as the d_in)
    context_lenght -> Number of tokens to process at the same time. (This affects the size of the query, key, value matrix)
    dropuot -> dropout percentage to apply 
    num_heads -> Number of heads to implement on the transformer block. (Each head output dim is d_out/num_heads)
    qkv_bias -> No use of the bias term for the Query, Key and Value matrix

  During tht forward pass the Attention mechanism is apply to the input, the output is the embeded vectors modified 
  based on the context and value of the other words.
"""
class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_length, context_length),
                    diagonal=1)
    )


  def forward(self, x):
    b, num_tokens, _ = x.shape

    keys = self.W_key(x)          # Shape: (b, num_tokens, d_out)
    queries = self.W_query(x)
    values = self.W_value(x)

    # Split the matrix on the heads 
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    # Attention scores 
    attn_scores = queries @ keys.transpose(2, 3)

    # Mask 
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)

    # Attention weights
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # Context
    context = (attn_weights @ values).transpose(1, 2) 

    # Combine all the heads 
    context = context.contiguous().view(b, num_tokens, self.d_out)
    context = self.out_proj(context)

    return context




"""
  GELU 
    Activation function on the Fully connected layer, this is the only non linear component
"""
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
        (x + 0.044715 * torch.pow(x, 3))
      )
    )



"""
  FeedForward
    Feed Forward with a GELU activation function, the the layer aplifies the emb_dim 4 times 
    and then reduces the dim to the output dim.
"""
class FeedForward(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(emb_dim, 4*emb_dim),
      GELU(),
      nn.Linear(4*emb_dim, emb_dim)
    )

  def forward(self, x):
    return self.layers(x)
  



"""
  LayerNorm
    Normalizaiton on the feature dimension, avoid overfiting
"""
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5  # Avoid division by 0
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.ones(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift





"""
  TransformerBlock
    Orchestrates the flow of the data inside the block, passing the data through all parts
"""
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # Components of the transformer block
    self.att = MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"], 
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"]
    )
    self.ff = FeedForward(cfg["emb_dim"])
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  # Flow of the data inside the transformer block
  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut    

    shortcut = x         
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut     
    return x



"""
  GPTModel
    The instance of the GPT model, instanciates and orchestrates all the parts of the GPT model.
    
    Input configuration: 
      GPT_CONFIG_124M = {
        "vocab_size":      # Vocabulary size             (Tokens)
        "context_length":  # Context length              (Num of words process at the time)
        "emb_dim":         # Embedding dimension         (Embedding)  
        "n_heads":         # Number of attention heads   (Attention)
        "n_layers":        # Number of layers            (MLP)
        "drop_rate":       # Dropout rate                (Regularization)
        "qkv_bias":        # Query-Key-Value bias        (No bias)
      }
"""
class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # Word embedding
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

    # Regularizaion
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    # Transformer
    self.trf_blocks = nn.Sequential(
      *[TransformerBlock(cfg) for I in range(cfg["n_layers"])]
    )

    # Layer normalization 
    self.final_norm = LayerNorm(cfg["emb_dim"])

    # UnEmbedding
    self.out_head = nn.Linear(
      cfg["emb_dim"], cfg["vocab_size"], bias=False
    )

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape

    # Process the embeddings 
    tok_embeds = self.tok_emb(in_idx)     # Work embedding 
    # The positional, if the seq_len is smaller than the context_length, we use the seq_len.. 
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds

    # Regularization
    x = self.drop_emb(x)

    # Transformer blocks 
    x = self.trf_blocks(x)

    # MLP
    x = self.final_norm(x)

    # Logits for the next token prediction
    logits = self.out_head(x)
    return logits


    