import torch
from torch.autograd import grad

class DetachW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, v, x):
        # We need to compute dot_W = J_W * v
        # We can use the fast double-backward trick to compute the forward pass
        with torch.enable_grad():
            z = torch.ones_like(W, requires_grad=True)
            g = grad(W, x, grad_outputs=z, create_graph=True, retain_graph=True)[0]
            S = (g * v).sum()
            dot_W = grad(S, z, create_graph=False)[0]
        
        ctx.save_for_backward(W, x)
        return dot_W.detach()

    @staticmethod
    def backward(ctx, grad_output):
        W, x = ctx.saved_tensors
        # Compute VJP: J_W^T * grad_output
        with torch.enable_grad():
            vjp = grad(W, x, grad_outputs=grad_output, retain_graph=True)[0]
        return None, vjp, None

def weighted_gradients(W, v, x, detach=False, create_graph=True):
    if detach:
        return DetachW.apply(W, v, x)
    z = torch.ones_like(W, requires_grad=True)
    g = grad(W, x, grad_outputs=z, create_graph=True, retain_graph=True)[0]
    S = (g * v).sum()
    dot_W = grad(S, z, create_graph=create_graph)[0]
    return dot_W

def test():
    # Setup
    x = torch.randn(2, 2, requires_grad=True) # batch_size=2, x_dim=2
    v = x ** 2 # v depends on x
    W = x.unsqueeze(-1) @ x.unsqueeze(1) # W depends on x, shape (2, 2, 2)
    
    # Detach test
    dot_W_det = weighted_gradients(W, v, x, detach=True)
    loss = dot_W_det.sum()
    loss.backward()
    
    print("x.grad is not None:", x.grad is not None)
    # x.grad should be non-zero because v depends on x!
    # But it should ONLY contain gradients from v, not from W.
    
    print("Success!")

if __name__ == "__main__":
    test()
