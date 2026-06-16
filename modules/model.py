import torch
import torch.nn as nn
import torchvision
import math

class ImprovedEfficientBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = self.efficientnet.features

    def forward(self, x):
        return self.features(x)  

class ImprovedPatchEmbedding(nn.Module):
    def __init__(self, in_channels=1280, embed_dim=384):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        """
        Input:  [B, 1280, 7, 7]
        Output: [B, 49, 384]
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  
        x = self.proj(x)  
        return x


class ImprovedViTBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class ImprovedEfficientViT(nn.Module):
    def __init__(self, embed_dim=384, depth=6, num_heads=4):
        super().__init__()
        self.backbone = ImprovedEfficientBackbone()
        self.patch_embed = ImprovedPatchEmbedding(embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_buffer("pos_embed", self._get_sinusoidal_encoding(50, embed_dim))

        self.patch_dropout = nn.Dropout(0.2)
        self.pos_dropout = nn.Dropout(0.2)

        self.blocks = nn.ModuleList([ImprovedViTBlock(embed_dim, num_heads) for _ in range(depth)])

        self.head = nn.Sequential(
              nn.LayerNorm(embed_dim),
              nn.Linear(embed_dim, 128),
              nn.GELU(),
              nn.Dropout(0.3),
              nn.Linear(128, 1)
          )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _get_sinusoidal_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  

    def forward(self, x):
        features = self.backbone(x)                  
        tokens = self.patch_embed(features)          
        tokens = self.patch_dropout(tokens)          

        B = tokens.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, tokens), dim=1)    
        x = x + self.pos_embed[:, :x.size(1), :]      
        x = self.pos_dropout(x)                       

        for block in self.blocks:
            x = block(x)

        cls_final = x[:, 0] 
        return self.head(cls_final)  