from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100


train_transforms = transforms.Compose([
    # implement data augmentation techniques here
    ...,
    ...,
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

val_transforms = transforms.Compose([
    # no data augmentation for validation set
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100(...)
val_dataset = CIFAR100(...)

class get_dataloaders():
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
        ''' Initialize the dataloader. '''

        # implement dataloader initialization here
        ...

    def get_train_loader(self):
        ''' Return the training dataloader. '''
        # implement training dataloader retrieval here
        ...
        return DataLoader(...)

    def get_val_loader(self):
        ''' Return the validation dataloader. '''
        # implement validation dataloader retrieval here
        ...
        return DataLoader(...)