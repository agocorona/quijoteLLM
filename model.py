import torch
import torch.nn as nn
from torch.nn import functional as F

# Transformer model implementation
# Implements a standard transformer architecture with:
# - Token and position embeddings
# - Multiple transformer encoder layers
# - Layer normalization and final linear projection
class TransformerModel(nn.Module):
    # Initialize transformer model
    # Args:
    #   vocab_size: Size of vocabulary
    #   n_embd: Embedding dimension (default: 512)
    #   n_head: Number of attention heads (default: 8)
    #   n_layer: Number of transformer layers (default: 6)
    #   block_size: Maximum sequence length (default: 128)
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
    # Forward pass through the transformer
    # Args:
    #   idx: Input token indices (batch_size, seq_len)
    # Returns: Logits for next token prediction (batch_size, seq_len, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    # Generate new tokens using the model
    # Args:
    #   idx: Initial token indices (batch_size, seq_len)
    #   max_new_tokens: Number of tokens to generate
    #   temperature: Sampling temperature (default: 1.0)
    #   top_k: Top-k sampling parameter (default: None)
    # Returns: Generated token sequence (batch_size, seq_len + max_new_tokens)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
