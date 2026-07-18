from dataclasses import dataclass

@dataclass
class model_config:
    num_classes: int = 100

    convnext_kernel_size : int = 7
    convnext_stride = 1

    stem_kernel_size : int = 3
    stem_stride : int = 1

    expansion_ratio : int = 4

    block1_channels : int = 96
    block2_channels : int = block1_channels * 2
    block3_channels : int = block1_channels * 4
    block4_channels : int = block1_channels * 8

    num_block1 : int = 3
    num_block2 : int = 3
    num_block3 : int = 9
    num_block4 : int = 3

    drop_path_rate : float = 0.2

@dataclass
class training_config:
    # dataloader parameters
    batch_size: int = 125
    num_workers: int = 2
    # training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 300
    weight_decay: float = 1e-1
    label_smoothing: float = 0.1