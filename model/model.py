'''
The Model architecture for CIFAR-100 classification task.
A ConvNext V2 is implemented.
'''
import torch
import torch.nn as nn
from .config import model_config
import torchinfo 

class layernorm(nn.Module):
    ''' 
    A seperate layernorm class is needed because pytorch has channels at dim = 1,
    while layernorm acts on dim = -1, hence we need to permute before applying layernorm
    and reverse the process'''
    def __init__(self,channels):
        '''
        Intialises the class
        Args:
            channels (int) : The channel count'''
        super().__init__()
        self.norm = nn.LayerNorm(channels,eps = 1e-6)

    def forward(self,x):
        '''
        Calls the forward function on x
        Args:
            x (tensor) : Input tensor
        Returns:
            x (tensor) : Normalised tensor
        '''
        # x = (B, C, H, W)
        x = x.permute(0,2,3,1)
        # now x = (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        # now x is again back to (B, C, H, W)
        return x 

class CNNStem(nn.Module):
    '''
    The Stem/First layer of the model
    
    This layer uses a large kernel size and stride to reduce spatial dims so that the
    deeper layers dont spend much compute.
    '''
    def __init__(self, config):
        ''' Initialises the class '''
        super().__init__()
        self.stem = nn.Conv2d(
            3,
            config.block1_channels,
            kernel_size = config.stem_kernel_size,
            stride = config.stem_stride,
            padding = 0
            )
        
        self.norm = layernorm(config.block1_channels)

    def forward(self,x):
        ''' 
        Calls forward on the input
        Args: 
            x (Tensor) : Input Tensor
        Returns:
            x (Tensor) : Output Tensor
        '''
        out = self.stem(x)
        out = self.norm(out)
        return out

class GRN(nn.Module):
    '''
    Global Response Normalization (GRN) introduced in ConvNeXt V2.

    GRN computes the global L2 response of each channel, normalizes it
    relative to the average response across all channels, and uses this
    information to rescale the activations. This encourages competition
    between channels, reduces feature redundancy, and improves feature
    diversity while preserving the original activations through a residual
    connection.
    '''
    def __init__(self, channels):
        '''
        Initialises the class
        Args:
            config (dataclass) : The hyperparam class
        '''
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self,x):
        '''
        Calls the forward func on x
        Args: 
            x (tensor) : Input Tensor
        Returns :
            x (tensor) : The Rescaled Tensor
        '''
        global_norm = torch.norm(x, dim = (2,3), p = 2, keepdim = True) 
        # p = 2 signifies the sqrt of sum of squared values over dim 2 and 3 (height and width)
        # basically gives a qrt of a scalar value of sum of squares of all the channels across their width and height
        # keep dim = True makes sure the dims dont collapse (we dont flatten)
        relative_norm = global_norm / (global_norm.mean(dim = 1, keepdim = True) + 1e-6)
        
        return self.gamma * (x * relative_norm) + self.beta + x

class Convnext(nn.Module):
    '''
    The Convnext block as described in the paper.
    '''
    def __init__(self, config, channels):
        '''
        Intialises the class
        Args:
            config (dataclass) : The hyperparam class
            channels (int) : No of input channels
        '''
        super().__init__()

        self.depth_conv = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = config.convnext_kernel_size,
            stride = config.convnext_stride,
            padding = "same",
            groups = channels # This is very important, this is a depth wise conv, meaning,
            # for groups zero, each out channel, operate on every in channel conv, but for groups = 2,
            # the out channels are grouped into 2 groups, each out channel group sees only one of the in channel group,
            # hence if groups = in_channels = out_channel like here, each out channel sees info from one in_channel and give out
            # the output. 
        )
        self.norm = layernorm(channels)
        self.expand = nn.Conv2d(
            in_channels = channels,
            out_channels = channels * config.expansion_ratio,
            kernel_size = 1,
            stride = 1,
        )
        self.grn = GRN(channels * config.expansion_ratio)
        self.activation = nn.GELU()
        self.shrink_back = nn.Conv2d(
            in_channels = channels * config.expansion_ratio,
            out_channels = channels,
            kernel_size = 1,
            stride = 1
        )

    def forward(self,x):
        '''
        Calls the forward function on the input tensor
        Args:
            x (tensor) : Input tensor
        Returns:
            x (tensor) : Output tensor'''
        out = self.depth_conv(x)
        out = self.norm(out)

        out = self.expand(out)
        out = self.activation(out)
        out = self.grn(out)
        out = self.shrink_back(out)

        out += x
        return out

class downsampler(nn.Module):
    '''
    The downsample that exists in between series of blocks,
    The deeper blocks get more channels but lesser spatial dims,
    this means the tensor becomes deeper in terms of channels but shrinks
    in width and height as it goes through the model
    '''
    def __init__(self, channels):
        '''
        Intitialises the class
        Args:
            channels (int) : No of input channels
        '''
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels = channels,
            out_channels = channels * 2,
            kernel_size = 2,
            stride = 2,
            padding = 0
        )

    def forward(self,x):
        '''
        Calls the forward function on the input tensor
        Args:
            x (tensor) : Input tensor
        Returns:    
            x (tensor) : The downsampled tensor
        '''
        return self.downsample(x)
    
class CIFAR100Model(nn.Module):
    '''
    ConvNext V2 model for CIFAR-100 classification task
    '''
    def __init__(self, config):
        '''
        Intialises the class
        Args:
            config (dataclass) : The hyperparam class
        '''
        super().__init__()
        self.stem = CNNStem(config)

        self.block1 = nn.ModuleList(
            Convnext(config,config.block1_channels) for _ in range(config.num_block1)
        )    

        self.downsampler1 = downsampler(config.block1_channels)

        self.block2 = nn.ModuleList(
            Convnext(config,config.block2_channels) for _ in range(config.num_block2)
        )    

        self.downsampler2 = downsampler(config.block2_channels)

        self.block3 = nn.ModuleList(
            Convnext(config,config.block3_channels) for _ in range(config.num_block3)
        )    

        self.downsampler3 = downsampler(config.block3_channels)

        self.block4 = nn.ModuleList(
            Convnext(config,config.block4_channels) for _ in range(config.num_block4)
        )    

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(config.block4_channels,config.hidden_features1),
            nn.GELU(),
            nn.Linear(config.hidden_features1, config.num_classes)
        )

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    

    def forward(self,x):
        '''
        Calls the forward function on the input tensor
        Args:
            x (tensor) : Input tensor
        Returns:
            x (tensor) : Output tensor
        '''
        x = self.stem(x)

        for block in self.block1:
            x = block(x)
        x = self.downsampler1(x)

        for block in self.block2:
            x = block(x)
        x = self.downsampler2(x)

        for block in self.block3:
            x = block(x)
        x = self.downsampler3(x)

        for block in self.block4:
            x = block(x)

        x = self.head(x)

        return x

if __name__ == "__main__":
    model = CIFAR100Model(config = model_config())
    torchinfo.summary(model, input_size=(1, 3, 32, 32))