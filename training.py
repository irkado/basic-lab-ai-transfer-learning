import argparse
import os
import time
from tempfile import TemporaryDirectory

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model import TransferModel
from preprocessing import train_loader, validation_loader, train_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


@torch.no_grad()
def eval_phase(model: nn.Module, loader, criterion: nn.Module):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += inputs.size(0)

    return (running_loss / total), (running_corrects / total)


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler=None,  # e.g. ReduceLROnPlateau
    num_epochs: int = 5,
    history_csv: str = "history.csv",
):
    since = time.time()

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += inputs.size(0)

            train_loss = running_loss / total
            train_acc = running_corrects / total

            val_loss, val_acc = eval_phase(model, validation_loader, criterion)

            # scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"lr {lr_now:.2e}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_params_path)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(lr_now)

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:.4f}")

        model.load_state_dict(torch.load(best_model_params_path, map_location=device))

    pd.DataFrame(history).to_csv(history_csv, index=False)
    print("Saved history to", history_csv)
    return model


def run_training(backbone: str, num_classes: int):
    print(f"\n=== Backbone: {backbone} ===")

    criterion = nn.CrossEntropyLoss()

    # head-only
    model = TransferModel(num_classes=num_classes, backbone=backbone, pretrained=True, dropout=0.2).to(device)
    model.train_head_only()

    optimizer = optim.Adam(model.get_trainable_params(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    print("training head-only")
    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=5,
        history_csv=f"{backbone}_head_history.csv",
    )
    torch.save(model.state_dict(), f"{backbone}_best_head.pt")

    # fine-tune last block
    model.load_state_dict(torch.load(f"{backbone}_best_head.pt", map_location=device))
    model.fine_tune_last_block()

    optimizer = optim.Adam(model.get_trainable_params(), lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    print("fine-tuning last block")
    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=7,
        history_csv=f"{backbone}_finetune_history.csv",
    )
    torch.save(model.state_dict(), f"{backbone}_best_finetuned.pt")


def main():
    num_classes = len(train_dataset.classes)
    print("num_classes:", num_classes)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["densenet121", "efficientnet_b0"],
        help="Which backbone to train",
    )
    args = parser.parse_args()

    run_training(args.backbone, num_classes)


if __name__ == "__main__":
    main()
