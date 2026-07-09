import torch
import torch.nn as nn
from config import model_config

class CIFAR100Model(nn.Module):
    def __init__(self, config):
        
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
        ''' Initialize the weights of the model. '''

        ...
        ...

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = CIFAR100Model(config = model_config())
    