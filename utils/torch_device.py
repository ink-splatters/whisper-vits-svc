import torch
import torch.backends.mps
import torch.cuda
from functools import cache


@cache
def cpu() -> torch.device:
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

@cache
def cuda(allow_fallback: bool = True, cuda_device_index: int | None = None) -> torch.device:        
    device: str | None =  'cuda' if cuda_device_index is None else f'cuda:{cuda_device_index}' if torch.cuda.is_available() else \
        cpu() if allow_fallback else None
    
    if device is None:
        raise RuntimeError("CUDA device is not available")

    return torch.device(device)