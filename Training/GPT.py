import torch
import torch.nn as nn




def generate_text_simple(model, idx, max_new_tokens, context_size): 
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]   
    with torch.no_grad():
        logits = model(idx_cond)

    logits = logits[:, -1, :]                   
    probas = torch.softmax(logits, dim=-1)         
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)   
    idx = torch.cat((idx, idx_next), dim=1)    

  return idx



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




# Transformer block (Multi-head attention and MLP)
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"], 
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"]
    )
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut      #2

    shortcut = x         #3
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut      #4
    return x

  
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module): 
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
      GELU(),
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
    )

  def forward(self, x):
    return x


# Normal layer - normalize the logits of the final output
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.ones(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift



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
  

