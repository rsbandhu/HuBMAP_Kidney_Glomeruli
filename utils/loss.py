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

def loss_focal_dice(y_pred, y_target, alpha, gamma,  ratio_focal_dice, reduction="mean"):

    assert ratio_focal_dice <= 1.0, "ration of dice loss to focal loss should be less than or equal to 1"
    assert ratio_focal_dice >= 0.0, "ration of dice loss to focal loss should be greater than or equal to 1"
    assert alpha <= 1.0, "alpha for focal loss should be less than or equal to 1"
    assert alpha <= 1.0, "alpha focal loss should be greater than or equal to 0"

    dice_loss = SoftDiceLoss( smooth=1.0 )
    dice = dice_loss(y_pred.sigmoid(), y_target)

    focal_loss = Focal_binary_loss(alpha, gamma, reduction)(y_pred, y_target)

    loss = focal_loss * ratio_focal_dice + (1.0- ratio_focal_dice)*dice

    return loss

class Focal_binary_loss(nn.Module):
    def __init__(self, alpha, gamma, reduction = None):
        super(Focal_binary_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = y_pred.sigmoid() #inputs are logits for binary mask
        pos_preds = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        neg_preds = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))

        pos_factor = self.alpha * (1-pos_preds)**self.gamma
        neg_factor = (1.0 -self.alpha)*(neg_preds**self.gamma)

        ce_pos = -pos_factor * torch.log(pos_preds)
        ce_neg = - neg_factor * torch.log(1-neg_preds)

        loss = ce_pos + ce_neg

        if self.reduction is None:
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()

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



