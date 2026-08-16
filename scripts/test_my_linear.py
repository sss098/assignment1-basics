import torch

from cs336_basics.model import Linear


layer = Linear(
    in_features=3,
    out_features=2,
)


print("Weight:")
print(layer.weights)

print("Weight shape:")
print(layer.weights.shape)


x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])


print("\nInput:")
print(x)

print("Input shape:")
print(x.shape)


y = layer(x)


print("\nOutput:")
print(y)

print("Output shape:")
print(y.shape)


x = torch.randn(
    4,
    10,
    3,
)

y = layer(x)

print(x.shape)  # (4, 10, 3)
print(y.shape)  # (4, 10, 2)