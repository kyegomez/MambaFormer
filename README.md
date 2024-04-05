[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MambaFormer
Implementation of MambaFormer in Pytorch ++ Zeta from the paper: "Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks"

## install
`pip3 install mamba-former`

## usage
```python
import torch
from mamba_former.main import MambaFormer

# Forward pass example
x = torch.randint(1, 1000, (1, 100))  # Token
# Tokens are integers representing input data

# Model
model = MambaFormer(
    dim=512,  # Dimension of the model
    num_tokens=1000,  # Number of unique tokens in the input data
    depth=6,  # Number of transformer layers
    d_state=512,  # Dimension of the transformer state
    d_conv=128,  # Dimension of the convolutional layer
    heads=8,  # Number of attention heads
    dim_head=64,  # Dimension of each attention head
    return_tokens=True,  # Whether to return the tokens in the output
)

# Forward pass
out = model(x)  # Perform a forward pass through the model

# If training
# out = model(x, return_loss=True)  # Perform a forward pass and calculate the loss

# Print the output
print(out)  # Print the output tensor
print(out.shape)  # Print the shape of the output tensor

```


# License
MIT
