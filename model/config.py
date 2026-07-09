from dataclasses import dataclass

@dataclass
class model_config:
    num_classes: int = 100
    conv1_out_channels: int = 32
    conv2_out_channels: int = 64
    kernel_size: int = 3


    pool_stride :int = 2
    pool_kernel :int = 2

    input_features:int = 4096 #64*8*8
    hidden_features1: int = 1028
    hidden_features2: int = 512

    dropout_rate : float = 0.3
    

@dataclass
class training_config:
    # dataloader parameters
    batch_size: int = 250
    num_workers: int = 4
    # training parameters
    learning_rate: float = 1e-2
    num_epochs: int = 100
    weight_decay: float = 1e-4