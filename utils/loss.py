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


# This loop is for callbacks. The loop can be changed to compute metrics. For example, if using class IoU(metric.Metric) replace the loss function with 
# these two lines: IoU_metric = IoU() then metric = IoU_metric.compute_metric(ypred)
for epoch in epochs:
    # put model in train mode
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        # zero the gradient
        optimizer.zero_grad()
        # get the data
        data = sample_batch["data"] # these two 'data' lines are defined depending on how the data was structured
        data = Variable(data) 
        # compute prediction
        ypred = model(data) 
        # compute loss
        loss = loss_function(ypred) # insert soft dice
        # backpropagate loss
        loss.backward()
        # apply backpropagation
        optimizer.step()
        # get the loss for this sample_batch
        loss.data.item()
        # add up loss over all the batches for a particular epoch
        epoch_loss += loss.data.item()

        # print statistics
        running_loss += loss.data.item() * images.size(0) # change images to whatever you named the output image as
        if i % 2000 == 1999:    # print every 2000 mini-batches. this is changed according to number of mini batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # print epoch loss
    print(epoch+1, epoch_loss) # if want average loss per epoch, do print(epoch+1, epoch_loss/len(training_set))

print('Finished Training')
