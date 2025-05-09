import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, plot_title=None, input_processor=None):
    train_losses = []
    val_losses = []
    model.to(device)

    for epoch in range(num_epochs):
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Epoch {epoch+1}/{num_epochs} | Start: [{start_time}]")

        # --- Training ---
        model.train()
        running_loss = 0.0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # print(f"{inputs.shape=}, {targets.shape=}")

            if input_processor is not None:
                inputs = input_processor(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        train_loss = running_loss / total
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}", end="")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if input_processor is not None:
                    inputs = input_processor(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                total += targets.size(0)

        val_loss /= total
        val_losses.append(val_loss)
        print(f"  Val Loss:   {val_loss:.4f}")

    print(f"best validation loss: {min(val_losses)}")
    print(f'{val_losses = }')
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if plot_title is None:
        plt.title('Training and Validation Loss')
    else:
        plt.title(plot_title)
    plt.legend()
    plt.show()