import tomllib
from datetime import datetime
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms

from models import uwcnn
from losses import CombinedLoss
from datasets import build_dataset
from training import train_model

if __name__ == '__main__':
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    dataset_root = cfg['dataset_root']
    dataset = build_dataset(dataset_root+"/raw-890", dataset_root+"/reference-890")
    
    dataset_len = len(dataset)
    train_size = int(0.7 * dataset_len)
    val_size = int(0.15 * dataset_len)
    test_size = dataset_len - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = uwcnn.UWCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = CombinedLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    train_model(model, train_loader, val_loader, optimizer, loss, device, num_epochs=10)
    