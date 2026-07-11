from dataclasses import dataclass

@dataclass
class model_config:
    num_classes: int = 100

    convnext_kernel_size : int = 7
    convnext_stride = 1

    stem_kernel_size : int = 4
    stem_stride : int = 4
    expansion_ratio : int = 4

    block1_channels : int = 96
    block2_channels : int = block1_channels * 2
    block3_channels : int = block1_channels * 4
    block4_channels : int = block1_channels * 8

    num_block1 : int = 3
    num_block2 : int = 3
    num_block3 : int = 9
    num_block4 : int = 3

    hidden_features1 : int = 1024
    hidden_features2 : int = 512

    dropout : float = 0.15
@dataclass
class training_config:
    # dataloader parameters
    batch_size: int = 250
    num_workers: int = 2
    # training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-1