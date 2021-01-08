import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        model_params = filter(lambda x: x.requires_grad, self.parameters())

        return super(BaseModel, self).__str__()

        