from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100


train_transforms = transforms.Compose([
    # implement data augmentation techniques here
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

val_transforms = transforms.Compose([
    # no data augmentation for validation set
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transforms)

class get_dataloaders():
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
        ''' Initialize the dataloader. '''

        # implement dataloader initialization here
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def get_train_loader(self):
        ''' Return the training dataloader. '''
        # implement training dataloader retrieval here
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def get_val_loader(self):
        ''' Return the validation dataloader. '''
        # implement validation dataloader retrieval here
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )