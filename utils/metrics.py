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



