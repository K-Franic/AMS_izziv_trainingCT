import torch
def device():
    return (torch.device("cuda" if torch.cuda.is_available() else "cpu"))