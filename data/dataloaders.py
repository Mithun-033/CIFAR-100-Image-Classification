'''
Implements the dataloaders for the CIFAR100 dataset.
Import the get_dataloaders class from this file in train_model.py to use the dataloaders.
'''
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100


train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transforms)

class get_dataloaders():
    def __init__(self, train_dataset = train_dataset, val_dataset = val_dataset, batch_size = 64, num_workers = 4):
        ''' 
        Initialize the dataloader. 
        Args:
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            batch_size: The batch size for the dataloader.
            num_workers: The number of workers for the dataloader.
        '''
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def get_train_loader(self):
        ''' 
        Implements the training dataloader.
        Returns:
            DataLoader: The training dataloader.
        '''
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers= True if self.num_workers > 0 else False,
            pin_memory= True
        )

    def get_val_loader(self):
        '''
        Implements the validation dataloader.
        Returns:
            DataLoader: The validation dataloader.
        '''
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory= True
        )

