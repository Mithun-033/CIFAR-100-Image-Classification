import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
from torchmetrics import Accuracy, Recall, F1Score, Precision

from rich import print
from rich.panel import Panel
from tqdm import tqdm
import json


from model.model import CIFAR100Model
from model.config import model_config, training_config
from data.dataloaders import get_dataloaders

torch.set_float32_matmul_precision("high")

def train_step(model, criterion, optimizer, train_loader, config, device, epoch, scheduler, ema_model):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    train_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch+1}/{config.num_epochs}]",
        
    )
    for images, labels in train_bar:
        images = images.to(device, non_blocking = True, memory_format = torch.channels_last)
        labels = labels.to(device, non_blocking = True)
        optimizer.zero_grad()
    
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        ema_model.update_parameters(model)
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_bar.set_postfix(
            loss=loss.item(),
            acc=f"{100*correct/total:.2f}%"
        )
    return running_loss / len(train_loader), 100 * correct / total

def eval_step(ema_model, criterion, val_loader, device):
    ema_model.eval()
    
    val_loss = 0.0
    acc = Accuracy(task="multiclass", num_classes=100).to(device)
    precision = Precision(task="multiclass", num_classes=100).to(device)
    recall = Recall(task="multiclass", num_classes=100).to(device)
    f1 = F1Score(task="multiclass", num_classes=100).to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking = True, memory_format = torch.channels_last)
            labels = labels.to(device, non_blocking = True)

            outputs = ema_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            acc.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            f1.update(outputs, labels)

    return val_loss / len(val_loader), acc.compute().item(), precision.compute().item(), recall.compute().item(), f1.compute().item()

def train_model(model: nn.Module, config: training_config):
    """
    Train the model.

    Args:
        model: CNN model
        config: Training configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device, memory_format = torch.channels_last)
    
    model = torch.compile(model)
    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(decay=0.999)
    )
    
    get_loaders = get_dataloaders(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader = get_loaders.get_train_loader()
    val_loader = get_loaders.get_val_loader()

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters= 0.05* len(train_loader) * config.num_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader) * 0.95,
        eta_min=config.learning_rate * 0.02
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[0.05 * len(train_loader) * config.num_epochs]
    )
    
    print(Panel.fit("Started Training ...", style="bold green"))

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_step(model, criterion, optimizer, train_loader, config, device, epoch, scheduler, ema_model)

        if (epoch + 1 )% 10 == 0:
            val_loss, val_acc, _, _, _ = eval_step(ema_model, criterion, val_loader, device)
            print(
                Panel.fit(
                    f"""
            Epoch {epoch+1}/{config.num_epochs}
    
            Train Loss : {train_loss:.4f}
            Train Acc  : {train_acc:.2f}%
    
            Val Loss   : {val_loss:.4f}
            Val Acc    : {val_acc:.2f}%
                            """,style="cyan"))
        
    torch.save(ema_model.state_dict(), "model.pt")
    print(Panel.fit("Training Complete and model saved!", style="bold green"))

if __name__ == "__main__":
    model = CIFAR100Model(model_config())
    summary(model, input_size=(1, 3, 32, 32))
    train_model(model, training_config())

