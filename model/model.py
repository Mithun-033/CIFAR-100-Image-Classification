''' Placeholder for the model architecture. '''

import torch.nn as nn
import torch.nn.functional as F
from model.config import model_config

class CIFAR100Model(nn.Module):
    def __init__(self, config):
        ''' Initialize the model. '''
        super().__init__()

        ...
        ...
        ...

        self.apply(self._init_weights)

    def _init_weights(self,module):
        ''' Initialize the weights of the model. '''

        ...
        ...

    def forward(self, x):
        ''' Forward pass of the model. '''

        ...
        ...

if __name__ == "__main__":
    model = CIFAR100Model(config = model_config())
    