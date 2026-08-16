import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
])

W = torch.nn.Parameter(
    torch.tensor(
    [0.1, 0.2, 0.3]
    )
)

print(x)
print(W)

print("x requires_grad:")
print(x.requires_grad)

print("w requires_grad:")
print(W.requires_grad)
