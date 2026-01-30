import os
import time
from tempfile import TemporaryDirectory
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import TransferModel
from preprocessing import train_loader, validation_loader, train_dataset, val_dataset  # <-- refactored imports

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def train_model(model, criterion, optimizer, num_epochs=5, history_csv="history.csv"):
    since = time.time()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                    loader = train_loader
                else:
                    model.eval()
                    loader = validation_loader

                running_loss = 0.0
                running_corrects = 0
                total = 0

                for inputs, labels in loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        preds = outputs.argmax(dim=1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum().item()
                    total += inputs.size(0)

                epoch_loss = running_loss / total
                epoch_acc = running_corrects / total

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "train":
                    train_loss, train_acc = epoch_loss, epoch_acc
                else:
                    val_loss, val_acc = epoch_loss, epoch_acc

                    # save best model on validation
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), best_model_params_path)

            # log after both phases
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:.4f}")

        model.load_state_dict(torch.load(best_model_params_path, map_location=device))

    pd.DataFrame(history).to_csv(history_csv, index=False)
    print("Saved history to", history_csv)

    return model


def main():
    num_classes = len(train_dataset.classes)
    print("num_classes:", num_classes)

    criterion = nn.CrossEntropyLoss()

    # head-only
    model = TransferModel(num_classes=num_classes, backbone="mobilenet_v2", pretrained=True, dropout=0.2).to(device)
    model.train_head_only()
    optimizer = optim.Adam(model.get_trainable_params(), lr=1e-3)

    print("\ntraining head-only")
    model = train_model(model, criterion, optimizer, num_epochs=5, history_csv="head_history.csv")
    torch.save(model.state_dict(), "best_head.pt")

    # fine-tune last block
    model.load_state_dict(torch.load("best_head.pt", map_location=device))
    model.fine_tune_last_block()
    optimizer = optim.Adam(model.get_trainable_params(), lr=1e-4)

    print("\nfine-tuning last block")
    model = train_model(model, criterion, optimizer, num_epochs=7, history_csv="lastblock_history.csv")
    torch.save(model.state_dict(), "best_lastblock.pt")


if __name__ == "__main__":
    main()
