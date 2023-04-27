import torch

a = torch.nn.Embedding(3,3)
a.requires_grad_(False)
print(a.weight.requires_grad)