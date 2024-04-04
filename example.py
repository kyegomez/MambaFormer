import torch 
from mamba_former.main import MambaFormer

# Forward pass example
x = torch.randint(1, 1000, (1, 100)) # Token
# Tokens are integrers

# Model
model = MambaFormer(
    dim = 512,
    num_tokens = 1000,
    depth = 6,
    d_state = 512,
    d_conv = 128,
    heads = 8,
    dim_head = 64,
    return_tokens = True
)

# Forward
out = model(x)
print(out)
print(out.shape)