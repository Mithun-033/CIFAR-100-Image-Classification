'''
The Model architecture for CIFAR-100 classification task.
A Convolutional Neural Network (CNN) is implemented.
'''
import torch
import torch.nn as nn
from .config import model_config
import torchinfo 

class CIFAR100Model(nn.Module):
    '''
    A Convolutional Neural Network (CNN) model for CIFAR-100 classification.
    The model consists of two convolutional layers followed by max pooling,
    and three fully connected layers for classification.
    '''
    def __init__(self, config):
        '''
        Initialize the CIFAR100Model with the given configuration.
        Args:
            config (model_config): Configuration parameters for the model.
        '''
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=config.conv1_out_channels,
                      kernel_size=config.kernel_size,
                      padding='same'
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(config.conv1_out_channels),
            nn.MaxPool2d(kernel_size = config.pool_kernel, stride = config.pool_stride), 

            nn.Conv2d(in_channels=config.conv1_out_channels,
                      out_channels=config.conv2_out_channels,
                      kernel_size=config.kernel_size,
                      padding='same'
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(config.conv2_out_channels),
            nn.MaxPool2d(kernel_size =config.pool_kernel, stride =config.pool_stride)
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_features , 
                    config.hidden_features1),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),


            nn.Linear(config.hidden_features1,
                    config.hidden_features2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),

            nn.Linear(config.hidden_features2 , config.num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self,module):
        '''
        Initialize the weights of the model using truncated normal distribution for Conv2d and Linear layers.
        This can be modified to use other initialization methods as needed.
        Args:
            module (nn.Module): The module to initialize.
        '''
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        '''
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        '''
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = CIFAR100Model(config = model_config())
    torchinfo.summary(model, input_size=(1, 3, 32, 32))