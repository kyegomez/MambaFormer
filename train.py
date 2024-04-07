from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-fp16")
#tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

print(len(tokenizer))


import torch 
from mamba_former.main import MambaFormer

# Forward pass example
x = torch.randint(1, 1000, (1, 100)) # Token
# Tokens are integrers

# Model
model = MambaFormer(
    dim = 128,
    num_tokens = len(tokenizer),
    depth = 2,
    d_state = 128,
    d_conv = 128,
    heads = 8,
    dim_head = 64,
    return_tokens = True
)

# Forward
out = model(x)
print(out)
print(out.shape)

# count parameters
model_size = sum(t.numel() for t in model.parameters())
print(f"parameter size: {model_size/1000**2:.1f}M parameters")


import datasets

data = datasets.load_dataset("roneneldan/TinyStories", split="train", num_proc=8)


from transformers import AutoTokenizer

def tokenize_function(examples):
    # Tokenize the texts
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="np")
    # Prepare inputs and labels by shifting the tokens
    result["labels"] = result["input_ids"][:, 1:].copy()
    result["input_ids"] = result["input_ids"][:, :-1].copy()
    return result

tokenized_dataset = data.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=8)


print(tokenized_dataset)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True)  # Adjust batch size as needed

from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True)  # Adjust batch size as needed

# Training Loop
model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids = torch.tensor(input_ids, device="cuda:0")
        attention_mask = torch.tensor(attention_mask, device="cuda:0")
        print('Type of input_ids:', type(input_ids))
        print('Shape of input_ids:', input_ids.shape)
        print('Type of attention_mask:', type(attention_mask))
        print('Shape of attention_mask:', attention_mask.shape)
        #labels = batch['labels']
        #outputs = model.forward(input_ids)
        #print(type(input_ids))
        labels = torch.ones_like(input_ids, device="cuda:0")
        inputs = torch.cat((input_ids, attention_mask, labels))
        inputs = torch.tensor(inputs, device="cuda:0")
        outputs = model(inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
