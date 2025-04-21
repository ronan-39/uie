import torch
from tqdm import tqdm
from datetime import datetime

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
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

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        train_loss = running_loss / total
        print(f"  Train Loss: {train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                total += targets.size(0)

        print(f"  Val Loss:   {val_loss:.4f}")