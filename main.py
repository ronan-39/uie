import tomllib
from datetime import datetime
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

from models import uwcnn, uwcnn_depth
from losses import CombinedLoss
from datasets import build_uied_dataset, build_depth_dataset
from training import train_model
from estimate_depth import RgbToRgbd

import matplotlib.pyplot as plt

def train_uwdcnn(model_name, dataset_name, cfg, gt_depth=True):
    print(f"Training UWDCNN-{model_name} on {dataset_name} for {cfg['num_epochs']} epochs. {gt_depth=}")

    uwdcnn_models = {
        'base': uwcnn_depth.UWCNN_Depth,
        'small': uwcnn_depth.Small_UWCNN_Depth,
        'micro': uwcnn_depth.Tiny_UWCNN_Depth,
        'nano': uwcnn_depth.Micro_UWCNN_Depth,
    }

    assert model_name in uwdcnn_models, "Specified model size does not exist"

    if gt_depth:
        assert 'depth' in cfg['datasets'][dataset_name], "Specified dataset does not have ground truth depth"
    
        dataset = build_depth_dataset(
            images_dir = cfg['datasets'][dataset_name]['img'],
            gt_dir = cfg['datasets'][dataset_name]['gt'],
            depth_dir = cfg['datasets'][dataset_name]['depth']
        )
    else:
        dataset = build_uied_dataset(
            images_dir = cfg['datasets'][dataset_name]['img'],
            labels_dir = cfg['datasets'][dataset_name]['gt'],
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = uwdcnn_models[model_name]()
    loss = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=cfg['betas'])
    
    if gt_depth:
        input_processor = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        input_processor = transforms.Compose([
            RgbToRgbd(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss,
        device,
        num_epochs=cfg['num_epochs'],
        plot_title=f'UWDCNN-{model_name} on {dataset_name}, {'gt depth' if gt_depth else 'estimated depth'}',
        input_processor=input_processor
    )

    evaluate_performance(model, test_loader, input_processor)

    return model


def train_uwcnn(model_name, dataset_name, cfg):
    print(f"Training UWCNN-{model_name} on {dataset_name} for {cfg['num_epochs']} epochs.")

    uwcnn_models = {
        'base': uwcnn.UWCNN,
        'small': uwcnn.Small_UWCNN,
        'micro': uwcnn.Tiny_UWCNN,
        'nano': uwcnn.Micro_UWCNN
    }

    assert model_name in uwcnn_models, "Specified model size does not exist"

    dataset = build_uied_dataset(
        images_dir = cfg['datasets'][dataset_name]['img'],
        labels_dir = cfg['datasets'][dataset_name]['gt'],
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = uwcnn_models[model_name]()
    loss = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=cfg['betas'])
    
    input_processor = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss,
        device,
        num_epochs=cfg['num_epochs'],
        plot_title=f'UWCNN-{model_name} on {dataset_name}',
        input_processor=input_processor
    )

    evaluate_performance(model, test_loader, input_processor)

    return model

@torch.no_grad()
def evaluate_performance(model, test_loader, input_processor):
    batch_psnrs = []
    batch_ssims = []

    device = next(model.parameters()).device

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = input_processor(inputs)
        predictions = model(inputs)
        batch_psnrs.append(psnr(predictions, targets).mean())
        batch_ssims.append(batch_ssim(predictions, targets).mean())

    print("Average PSNR on test set:", round(torch.Tensor(batch_psnrs).mean().item(), 4))
    print("Average SSIM on test set:", round(torch.Tensor(batch_ssims).mean().item(), 4))

def psnr(prediction, target, max_val=1):
    mse = F.mse_loss(prediction, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # mean per image
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr

def batch_ssim(prediction, target, data_range=None):
    prediction = prediction.permute(0,2,3,1).cpu().numpy()
    target = target.permute(0,2,3,1).cpu().numpy()
    assert prediction.shape == target.shape, "Input batches must have the same shape"
    batch_size = prediction.shape[0]
    multichannel = prediction.ndim == 4  # (N, H, W, C)

    ssim_scores = []
    for i in range(batch_size):
        range_val = data_range
        if data_range is None:
            range_val = max(prediction[i].max(), target[i].max())
        score = ssim(prediction[i], target[i], data_range=range_val, channel_axis=-1 if multichannel else None)
        ssim_scores.append(score)

    return np.array(ssim_scores)

def main():
    '''
    train models
    '''
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    # available model names:
    # base, small, micro

    def t1():
        '''
        no depth
        '''
        model_name = 'base'
        dataset_name = 'nyu_type3'
        model = train_uwcnn(model_name, dataset_name, cfg)
        print(f"Trained UWCNN-{model_name} on {dataset_name} for {cfg['num_epochs']} epochs.")
        torch.save(model.state_dict(), f'./checkpoints/uwcnn-{model_name}_{dataset_name}_{cfg['num_epochs']}_epoch.pth')


    def t2():
        '''
        depth
        '''
        model_name = 'base'
        dataset_name = 'nyu_type3'
        gt_depth = False
        model = train_uwdcnn(model_name, dataset_name, cfg, gt_depth=gt_depth)
        print(f"Trained UWDCNN-{model_name} on {dataset_name} for {cfg['num_epochs']} epochs. {gt_depth=}")
        torch.save(model.state_dict(), f'./checkpoints/uwdcnn-{model_name}_{dataset_name}_{cfg['num_epochs']}_epoch_{'gt' if gt_depth else 'est'}.pth')

    t1()
    torch.cuda.empty_cache()
    t2()

def view_model_sample_outputs():
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    model = uwcnn_depth.UWCNN_Depth()
    checkpoint = torch.load('./checkpoints/uwdcnn-base_nyu_type3_50_epoch.pth')
    model.load_state_dict(checkpoint)

    input_processor = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset_name = 'nyu_type3'
    dataset = build_depth_dataset(
        images_dir = cfg['datasets'][dataset_name]['img'],
        gt_dir = cfg['datasets'][dataset_name]['gt'],
        depth_dir = cfg['datasets'][dataset_name]['depth']
    )

    img, label = dataset[1]
    input = input_processor(img.unsqueeze(0))

    with torch.no_grad():
        prediction = model(input)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    axs[0].imshow(img[:-1].permute(1,2,0))
    axs[0].set_title("input")
    
    axs[1].imshow(label.permute(1,2,0))
    axs[1].set_title("ground truth")

    axs[2].imshow(prediction[0].permute(1,2,0))
    axs[2].set_title("prediction")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # main()

    '''
        eval ssim metric on the trained models
    '''
    # with open('./config.toml', 'rb') as f:
    #     cfg = tomllib.load(f)

    # # model = uwcnn_depth.UWCNN_Depth()
    # # checkpoint = torch.load('./checkpoints/uwdcnn-base_nyu_type3.pth')
    # model = uwcnn.Tiny_UWCNN()
    # checkpoint = torch.load('./checkpoints/uwcnn-micro_nyu_type3.pth')
    # model.load_state_dict(checkpoint)

    # input_processor = transforms.Compose([
    #     transforms.Normalize((0.5,), (0.5,)),
    # ])

    # dataset_name = 'nyu_type3'
    # dataset = build_uied_dataset(
    #     images_dir = cfg['datasets'][dataset_name]['img'],
    #     labels_dir = cfg['datasets'][dataset_name]['gt'],
    #     # depth_dir = cfg['datasets'][dataset_name]['depth']
    # )

    # dataset_len = len(dataset)
    # train_size = int(cfg['train_split'] * dataset_len)
    # val_size = int(cfg['val_split'] * dataset_len)
    # test_size = dataset_len - train_size - val_size

    # _, _, test_dataset = random_split(
    #     dataset,
    #     [train_size, val_size, test_size],
    #     generator=torch.Generator().manual_seed(1)
    # )

    # test_loader = DataLoader(test_dataset, batch_size=32)

    # evaluate_performance(model, test_loader, input_processor)
