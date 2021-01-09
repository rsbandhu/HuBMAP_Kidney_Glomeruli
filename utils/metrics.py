import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.sum = None
        self.count = None
        self.avg = None
        self.initialized = False

    def initialize(self, val, weight):
        self.val = val
        self.count = weight
        self.sum = np.multiply(val, weight)
        self.avg = val
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += (val*weight)
        self.count += weight
        self.avg = self.sum / self.count

        
import torch
import numpy as np
from metric import metric
from metric.confusionmatrix import ConfusionMatrix
'''Imports needed to compute IoU metric'''

class IoU(metric.Metric):
    
    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None: # let ignore_index be None
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")
                
    def reset(self):
        self.conf_metric.reset()
        
    def add(self, predicted, target): # adds the predicted and target pairs for IoU computation
        # predicted and target: Tensor of shape (N, K, H, W) or (N, H, W) of int values between 0 and K - 1
        # where K is number of classes, N is num of samples
        
        # this block of code chekcs the dimensions to make sure everything matches up 
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"
        
        self.conf_metric.add(predicted.view(-1), target.view(-1)) # .view reshapes the output tensor according to your desired output 
      
    def compute_metric(self):
        '''returns a tuple where the first ouput is the per class IoU (binary in our case) 
            and the second output is mean IoU'''
        
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive
         
        with np.errstate(divide='ignore', invalid='ignore'): # ignores divide by 0 errors
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)
    
'''ADD PREDICTED and TARGET tensor shapes. '''
pred_shape = torch.tensor([N, '''1 or 3?''', 512, 512])
targ_shape = torch.tensor([N, 1, 512, 512])
    
IoU_metric = IoU() # metric instance to use during training
IoU_metric.add(pred_shape, targ_shape) # instance to inherit add function. edit shapes accordingly to input dims
IoU_metric.compute_metric() 

import time
'''Return metric after each epoch. Still prototyping the code, but it runs without any network implemented just yet.'''

def compute_metric(callback=None):   
        if callback: callback.before_compute(i) 
        metric = IoU_metric.compute_metric()
        print("IoU score: {}".format(metric))
        time.sleep(1)
        if callback: callback.after_compute(i, val=metric)        
    return metric     

class PrintCallback():
    def __init__(self): pass
    def before_compute(self, epoch): print(f"Epoch begin: {epoch}")
    def after_compute(self, epoch): print(f"Epoch end: {epoch}: {val}")
