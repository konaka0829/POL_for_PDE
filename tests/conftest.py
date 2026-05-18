try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass
