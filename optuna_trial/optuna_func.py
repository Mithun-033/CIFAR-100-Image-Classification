import optuna 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from model.model import CIFAR100Model
from model.config import model_config, training_config
from data.dataloaders import get_dataloaders
from train.train_model import train_step, eval_step
from rich import print, panel

global_best_val_acc = 0.0

def objective(trial : optuna.Trial):
    num_block1 = trial.suggest_int("num_block1", 2, 4)
    num_block2 = trial.suggest_int("num_block2", 2, 4)
    num_block3 = trial.suggest_int("num_block3", 8, 12)
    num_block4 = trial.suggest_int("num_block4", 2, 4)


    expansion_ratio = trial.suggest_int("expansion_ratio", 3, 5)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.1, 0.3)

    block1_channels = trial.suggest_int("block1_channels", 64, 128)
    block2_channels = block1_channels * 2
    block3_channels = block1_channels * 4
    block4_channels = block1_channels * 8

    optuna_model_config = model_config(
        num_block1 = num_block1,
        num_block2 = num_block2,
        num_block3 = num_block3,
        num_block4 = num_block4,
        expansion_ratio = expansion_ratio,
        drop_path_rate = drop_path_rate,
        block1_channels = block1_channels,
        block2_channels = block2_channels,
        block3_channels = block3_channels,
        block4_channels = block4_channels
    )

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.2)
    
    optuna_training_config = training_config(
        learning_rate = lr,
        weight_decay = weight_decay,
        label_smoothing = label_smoothing
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR100Model(optuna_model_config).to(device, memory_format = torch.channels_last)
    model = torch.compile(model)
    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(decay=0.999)
    ).to(device, memory_format = torch.channels_last)


    criterion = nn.CrossEntropyLoss(label_smoothing=optuna_training_config.label_smoothing)
    get_loaders = get_dataloaders(batch_size=optuna_training_config.batch_size, num_workers=optuna_training_config.num_workers)
    train_loader = get_loaders.get_train_loader()
    val_loader = get_loaders.get_val_loader()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optuna_training_config.learning_rate,
        weight_decay=optuna_training_config.weight_decay
    )
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters= int(0.05* len(train_loader) * optuna_training_config.num_epochs)
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(optuna_training_config.num_epochs * len(train_loader) * 0.95),
        eta_min=optuna_training_config.learning_rate * 0.05
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(0.05 * len(train_loader) * optuna_training_config.num_epochs)]
    )
    global global_best_val_acc
    print(
    panel.Panel(
        f"Trial {trial.number}\n"
        f"lr = {lr:.6f}\n"
        f"weight_decay = {weight_decay:.6f}\n"
        f"num_block1 = {num_block1}\n"
        f"num_block2 = {num_block2}\n"
        f"num_block3 = {num_block3}\n"
        f"num_block4 = {num_block4}\n"
        f"expansion_ratio = {expansion_ratio}\n"
        f"drop_path_rate = {drop_path_rate:.4f}\n",
        title=f"Trial {trial.number}",
        style="cyan",
        expand=False,
    )
    )   
    params = 0
    for param in model.parameters():
        params += param.numel()
        
    print(panel.Panel(f"Total Parameters: {params :_}", style="yellow"))    
    print(panel.Panel("Training started...", style="bold green"))

    for epoch in range(optuna_training_config.num_epochs):
        train_loss, train_acc = train_step(model, criterion, optimizer, train_loader, optuna_training_config, device, epoch, scheduler, ema_model)
        val_loss, val_acc, pre, rec, f1 = eval_step(ema_model, criterion, val_loader, device)

        if val_acc > global_best_val_acc:
            torch.save(ema_model.state_dict(), "best_model.pt")
            global_best_val_acc = val_acc
        trial.report(val_acc, epoch + 1)

        if trial.should_prune():
            trial.set_user_attr("train_loss", train_loss)
            trial.set_user_attr("train_acc", train_acc)
            trial.set_user_attr("val_loss", val_loss)
            trial.set_user_attr("val_acc", val_acc)
            trial.set_user_attr("precision", pre)
            trial.set_user_attr("recall", rec)
            trial.set_user_attr("f1_score", f1)
            raise optuna.exceptions.TrialPruned()
        
    trial.set_user_attr("train_loss", train_loss)
    trial.set_user_attr("train_acc", train_acc)
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("val_acc", val_acc)
    trial.set_user_attr("precision", pre)
    trial.set_user_attr("recall", rec)
    trial.set_user_attr("f1_score", f1)
    return val_acc

if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10,
        max_resource=100,
        reduction_factor=3
    )
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage="sqlite:///optuna_study.db",
        study_name="cifar100_optuna_study",
    )
    study.optimize(objective, n_trials=50)