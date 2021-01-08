import torch.nn as nn
import torch.optim

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth, dims=(-2,-1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, output, target):
        
        tp = (output*target).sum(self.dims)
        
        fp = (output*(1.0-target)).sum(self.dims)
        
        fn = ((1.0-output)*target).sum(self.dims)

        dc = (2.0*tp + self.smooth) / (2*tp + fp + fn + self.smooth)

        dc = dc.mean()

        return 1- dc

cross_ent_loss = nn.BCEWithLogitsLoss()

dice_loss = SoftDiceLoss(smooth=1.0)

def loss_fn(y_pred, y_target, ratio_dice_bce):

    bce = cross_ent_loss(y_pred, y_target)
    dice = dice_loss(y_pred.sigmoid(), y_target)

    loss = bce*ratio_dice_bce + (1.0 - ratio_dice_bce)*dice

    return loss

def use_optimizer(network, config):
    optim_params = config["optimizer"]
    if optim_params["type"] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr= optim_params["sgd"]["lr"],
                                    momentum= optim_params["sgd"]["momentum"],
                                    weight_decay=optim_params["sgd"]["weight_decay"])
    elif optim_params["type"] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=optim_params["adam"]["lr"],
                                     weight_decay=optim_params["sgd"]["weight_decay"])
    return optimizer



