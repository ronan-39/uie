import tomllib
from datetime import datetime
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

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
    
    if gt_depth == False:
        raise NotImplementedError("Can't predict depth yet")

    dataset = build_depth_dataset(
        images_dir = cfg['datasets'][dataset_name]['img'],
        gt_dir = cfg['datasets'][dataset_name]['gt'],
        depth_dir = cfg['datasets'][dataset_name]['depth']
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
        gt_dir = cfg['datasets'][dataset_name]['gt'],
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

    device = next(model.parameters()).device

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = input_processor(inputs)
        predictions = model(inputs)
        batch_psnrs.append(psnr(predictions, targets).mean())

    print("Average PSNR on test set:", round(torch.Tensor(batch_psnrs).mean().item(), 4))

def psnr(prediction, target, max_val=1):
    mse = F.mse_loss(prediction, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # mean per image
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr

if __name__ == '__main__':
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    def test1():
        '''
        no depth
        '''
        model_name = 'base'
        dataset_name = 'nyu_type3'
        model = train_uwcnn(model_name, dataset_name, cfg)
        torch.save(model.state_dict(), f'./checkpoints/uwcnn-{model_name}_{dataset_name}.pth')


    def test2():
        '''
        depth
        '''
        model_name = 'micro'
        dataset_name = 'nyu_type3'
        model = train_uwdcnn(model_name, dataset_name, cfg, gt_depth=True)
        torch.save(model.state_dict(), f'./checkpoints/uwdcnn-{model_name}_{dataset_name}.pth')

    test2()
