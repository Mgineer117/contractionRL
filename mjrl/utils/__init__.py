"""mjrl utilities (sampler, runner)."""
import torch

def get_device() -> str:
    """Detect device and prioritize cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
