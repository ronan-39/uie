import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torch

class UIEDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        self.image_filenames = sorted(os.listdir(images_dir))
        self.label_filenames = sorted(os.listdir(labels_dir))
        assert self.image_filenames == self.label_filenames, "Image-label mismatch!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

def build_dataset(images_dir, labels_dir):
    transform = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToPureTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return UIEDataset(images_dir, labels_dir, transform)