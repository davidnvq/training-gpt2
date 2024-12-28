import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # ! optimized step 1: causal mask on-the-fly to reduce memory usage
        # self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # shape: (B, H, L, D]
        k = self.k_proj(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # shape: (B, H, L, L)
        attn_scores = q @ k.transpose(-2, -1)

        # ! optimized step 1: causal mask on-the-fly to reduce memory usage
        mask_bool = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device), diagonal=1).bool()

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # ! optimized step 6: use FlashAttention v2 using nn.functional.scaled_dot_product_attention
        context_vec = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_bool, dropout_p=self.dropout.p, is_causal=True)

        # shape: (B, L, D)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class FeedForward(nn.Module):

    def __init__(self, emb_dim, d_feedforward):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, d_feedforward)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_feedforward, emb_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias,
        )
        self.ffn = FeedForward(config.emb_dim, config.emb_dim * 4)
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ffn(self.norm2(x)))
        return x


class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)

        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])

        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
