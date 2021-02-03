import torch
import torch.nn as nn


def load_state():
    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url


def initialize_weights(*models):
    # type: (models) -> None
    for model in models:
        for m in model.modules():
            if isinstance( m, nn.Conv2d ):
                nn.init.kaiming_normal_( m.weight.data, nonlinearity='relu')
            elif isinstance( m, nn.BatchNorm2d ):
                m.weight.data.fill_( 1. )
                m.bias.data.fill_(1e-4 )
            elif isinstance( m, nn.Linear ):
                m.weight.data.normal_( 0.0, 0.0001 )
                m.bias.data.zero_()
