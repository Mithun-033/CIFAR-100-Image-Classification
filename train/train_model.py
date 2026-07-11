import torch
import torch.nn as nn
import torch.optim as optim
from rich import print
from rich.panel import Panel
from tqdm import tqdm
import json
from torchinfo import summary

from model.model import CIFAR100Model
from model.config import model_config, training_config
from data.dataloaders import get_dataloaders

torch.set_float32_matmul_precision("high")

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

    get_loaders = get_dataloaders(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader = get_loaders.get_train_loader()
    val_loader = get_loaders.get_val_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader),
        eta_min=config.learning_rate * 0.05
    )
    train_loss_lst = []
    val_loss_lst = []
    print(Panel.fit("Started Training ...", style="bold green"))

    for epoch in range(config.num_epochs):

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
            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(
                loss=loss.item(),
                acc=f"{100*correct/total:.2f}%"
            )

        train_loss = running_loss / len(train_loader)
        train_loss_lst.append(train_loss)
        train_acc = 100 * correct / total

        if (epoch + 1 )% 10 == 0:
            model.eval()
    
            val_loss = 0.0
            val_correct = 0
            val_total = 0
    
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking = True, memory_format = torch.channels_last)
                    labels = labels.to(device, non_blocking = True)
    
                    outputs = model(images)
    
                    loss = criterion(outputs, labels)
    
                    val_loss += loss.item()
    
                    _, predicted = outputs.max(1)
    
                    val_total += labels.size(0)
    
                    val_correct += predicted.eq(labels).sum().item()
    
            val_loss /= len(val_loader)
            val_loss_lst.append(val_loss)
            val_acc = 100 * val_correct / val_total
    
            print(
                Panel.fit(
                    f"""
            Epoch {epoch+1}/{config.num_epochs}
    
            Train Loss : {train_loss:.4f}
            Train Acc  : {train_acc:.2f}%
    
            Val Loss   : {val_loss:.4f}
            Val Acc    : {val_acc:.2f}%
                            """,style="cyan"))
        
    with open("train_loss.json", "w") as f:
        json.dump(train_loss_lst, f)
    with open("val_loss.json", "w") as f:
        json.dump(val_loss_lst, f)

    torch.save(model.state_dict(), "model.pt")
    print(Panel.fit("Training Complete and model saved!", style="bold green"))

if __name__ == "__main__":
    model = CIFAR100Model(model_config())
    summary(model, input_size=(1, 3, 32, 32))
    train_model(model, training_config())

