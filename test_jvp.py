import torch
from torch.autograd import grad

torch.manual_seed(42)
n, d, x_dim = 2, 3, 4
x = torch.randn(n, x_dim, requires_grad=True)
v = torch.randn(n, x_dim, requires_grad=True)

# Some function W(x)
# W_{ij} = sum_k x_k * (i+1) * (j+1) + x_0**2
W = torch.zeros(n, d, d, dtype=x.dtype, device=x.device)
for i in range(d):
    for j in range(d):
        W[:, i, j] = x.sum(dim=-1) * (i+1) * (j+1) + x[:, 0]**2

# Method 1: Original loop
def original(W, v, x):
    J = torch.zeros(n, d, d, x_dim)
    for i in range(d):
        for j in range(d):
            J[:, i, j, :] = grad(W[:, i, j].sum(), x, create_graph=True, retain_graph=True)[0]
    return (J * v.view(n, 1, 1, -1)).sum(dim=3)

# Method 2: Double backward trick
def new_method(W, v, x):
    z = torch.ones_like(W, requires_grad=True)
    g = grad(W, x, grad_outputs=z, create_graph=True, retain_graph=True)[0]
    S = (g * v).sum()
    dot_W = grad(S, z, create_graph=True)[0]
    return dot_W

out1 = original(W, v, x)
out2 = new_method(W, v, x)

print("Max diff:", (out1 - out2).abs().max().item())
print("Requires grad:", out2.requires_grad)

# Test backward through out2
loss = out2.sum()
loss.backward()
print("x grad:", x.grad)
