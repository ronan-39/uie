import tomllib
from datetime import datetime
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms

from models import uwcnn, uwcnn_depth
from losses import CombinedLoss
from datasets import build_uied_dataset, build_nyu_dataset
from training import train_model

import matplotlib.pyplot as plt

def train_uied():
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    dataset_root = cfg['uieb_dataset_root']
    dataset = build_uied_dataset(dataset_root+"/raw-890", dataset_root+"/reference-890")

    dataset_len = len(dataset)
    train_size = int(cfg['train_split'] * dataset_len)
    val_size = int(cfg['val_split'] * dataset_len)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=cfg['betas'])

    train_model(model, train_loader, val_loader, optimizer, loss, device, num_epochs=cfg['num_epochs'], plot_title="UIED Training")

def train_nyu_with_depth():
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    dataset_root = cfg['nyu_dataset_root']
    dataset = build_nyu_dataset(
        images_dir=dataset_root+"/type3_data/underwater_type_3",
        depth_maps_dir=dataset_root+"/type3_data/transmission_type_3",
        labels_dir=dataset_root+"/type3_data/gt_type_type_3"
    )

    # im, _ = dataset[1]
    # print(im.shape)
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].imshow(im[:3].permute(1,2,0))
    # axim = axs[1].imshow(im[-1])
    # fig.colorbar(axim, ax=axs[1], label='Pixel value')
    # print(im[-1].max(), im[-1].min())
    # plt.show()
    # import sys
    # sys.exit()

    dataset_len = len(dataset)
    train_size = int(cfg['train_split'] * dataset_len)
    val_size = int(cfg['val_split'] * dataset_len)
    test_size = dataset_len - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = uwcnn_depth.UWCNN_Depth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = CombinedLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=cfg['betas'])

    train_model(model, train_loader, val_loader, optimizer, loss, device, num_epochs=cfg['num_epochs'], plot_title="NYU with Depth Training")
    print("NYU with depth")
    eval(model, test_loader, loss, device)
    test_img, test_label = test_dataset[0]
    with torch.no_grad():
        output = model(test_img.unsqueeze(0).to(device)).cpu()[0]
    print(f'{output.shape = }')
    rgb = test_img[0:3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    fig, axs = plt.subplots(ncols=3)
    axs[0].imshow(rgb.permute(1,2,0))
    axs[0].set_title("input")

    axs[1].imshow(output.permute(1,2,0))
    axs[1].set_title("prediction")

    axs[2].imshow(test_label.permute(1,2,0))
    axs[2].set_title("ground truth")
    plt.show()

def train_nyu_no_depth():
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    dataset_root = cfg['nyu_dataset_root']
    dataset = build_uied_dataset(
        images_dir=dataset_root+"/type3_data/underwater_type_3",
        labels_dir=dataset_root+"/type3_data/gt_type_type_3"
    )

    dataset_len = len(dataset)
    train_size = int(cfg['train_split'] * dataset_len)
    val_size = int(cfg['val_split'] * dataset_len)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=cfg['betas'])

    train_model(model, train_loader, val_loader, optimizer, loss, device, num_epochs=cfg['num_epochs'], plot_title="NYU with no Depth Training, Tiny Model")
    print("NYU no depth, tiny model")
    eval(model, test_loader, loss, device)
    print(cfg)

    test_img, test_label = test_dataset[0]
    rgb = test_img[0:3]
    with torch.no_grad():
        output = model(rgb.unsqueeze(0).to(device)).cpu()[0]
    print(f'{output.shape = }')
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    fig, axs = plt.subplots(ncols=3)
    axs[0].imshow(rgb.permute(1,2,0))
    axs[0].set_title("input")

    axs[1].imshow(output.permute(1,2,0))
    axs[1].set_title("prediction")

    axs[2].imshow(test_label.permute(1,2,0))
    axs[2].set_title("ground truth")
    plt.show()

def eval(model, test_loader, criterion, device):
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # no gradient tracking for evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1

    average_test_loss = total_loss / num_batches
    print(f"Average test loss: {average_test_loss:.4f}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num params: {num_params}")


if __name__ == '__main__':
    train_nyu_with_depth()
    # train_nyu_no_depth()

    # with open('./config.toml', 'rb') as f:
    #     cfg = tomllib.load(f)

    # dataset_root = cfg['nyu_dataset_root']
    # dataset = build_nyu_dataset(
    #     images_dir=dataset_root+"/type3_data/underwater_type_3",
    #     depth_maps_dir=dataset_root+"/type3_data/transmission_type_3",
    #     labels_dir=dataset_root+"/type3_data/gt_type_type_3"
    # )

    # im, label = dataset[3]
    # rgb = im[0:3]
    # rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    # depth = im[3]
    # fig, axs = plt.subplots(ncols=3)
    # axs[0].imshow(rgb.permute(1,2,0))
    # axs[0].set_title("image")
    # axs[1].imshow(label[:3].permute(1,2,0))
    # axs[1].set_title("label")
    # axim = axs[2].imshow(depth, cmap='gray')
    # fig.colorbar(axim, ax=axs[2], label='Pixel value')
    # axs[2].set_title("depth map")
    # print(depth.max(), depth.min())
    # plt.show()