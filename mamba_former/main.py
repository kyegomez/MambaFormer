import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock, OutputHead
from zeta.nn.attention.multiquery_attention import MultiQueryAttention

class MambaFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        heads: int,
        dim_head: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.dim_head = dim_head
        
        # MambaBlock
        self.input_mamba = MambaBlock(
            dim,
            1,
            d_state,
            d_conv,
            *args,
            **kwargs
        )
        
        # MultiQueryAttention
        self.attn = MultiQueryAttention(
            dim,
            heads,
            *args,
            **kwargs   
        )
    
    def forward(self, x: Tensor):
        # First step is Attention
        skip = x
        
        # Attention
        attended, _, _ = self.attn(x)
        
        # Add residual
        x = attended + skip
        
        # Second step is MambaBlock
        skip_two = x
        
        # Mamba
        x = self.input_mamba(x)
        
        # Add residual and return
        return x + skip_two
    

class MambaFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        depth: int,
        d_state: int,
        d_conv: int,
        heads: int,
        dim_head: int,
        return_tokens: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.num_tokens = num_tokens
        self.depth = depth
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.dim_head = dim_head
        self.return_tokens = return_tokens
        
        # Embedding
        self.embed = nn.Embedding(num_tokens, dim)
        
        # Mamba as input
        self.input_mamba = MambaBlock(
            dim,
            1,
            d_state,
            d_conv,
            *args,
            **kwargs
        )
        
        # Layers of Mamba Former Block
        self.layers = nn.ModuleList([
            MambaFormerBlock(
                dim,
                d_state,
                d_conv,
                heads,
                dim_head,
            )
            for _ in range(depth)
        ])
        
        
        # Norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Embed tokens to get tensors
        x = self.embed(x)
        
        # Normalize
        x = self.norm(x)
        
        # Then skip
        skip = x
        
        # Mamba
        x = self.input_mamba(x) + skip
        
        # MambaFormer Blocks
        for layer in self.layers:
            x = layer(x) + x
        
        if self.return_tokens:
            out = OutputHead(self.dim, -1)(x)
            return out
        else: 
            return x
        
