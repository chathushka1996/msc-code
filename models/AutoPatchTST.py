import torch
import torch.nn as nn
import torch.optim as optim

# === Model Components ===
class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_size, embed_dim):
        super().__init__()
        print(embed_dim)
        self.proj = nn.Linear(patch_size * input_dim, embed_dim)
        self.patch_size = patch_size
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels
        print(B, T, C)
        x = x.unfold(1, self.patch_size, self.patch_size).contiguous()
        print(x.size())
        x = x.view(B, -1, self.patch_size * C)
        print(x.size())
        x = self.proj(x)
        return x

class AutoformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = 8
        patch_size = 16
        embed_dim = 128
        num_heads = 4
        ffn_dim = 256
        depth = 4
        output_len = 96  # Can be varied
        self.embed = PatchEmbedding(input_dim, patch_size, embed_dim)
        self.encoder = nn.Sequential(*[AutoformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, output_len)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))
        return x