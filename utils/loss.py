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


# To view loss after each epoch 
num_epochs = ...
n_batches = ...
for epoch in range(2): # loop over the dataset multiple times
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): # in enumerate('''number of tile images'''):
        # get the inputs
        inputs, labels = data # here would be sample_batches

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) # change loss to soft dice
        loss.backward()
        optimizer.step() 

        epoch_loss += outputs.shape[0] * loss.item()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches. THIS WOULD BE CHANGED ACCORDINGLY
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # print epoch loss
    print(epoch+1, epoch_loss / len(trainset # change to name of train set))

print('Finished Training')
