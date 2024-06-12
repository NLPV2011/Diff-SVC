import torch
import torch.nn.functional as F

def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)
