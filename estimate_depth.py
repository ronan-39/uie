from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import tomllib
from datasets import build_uied_dataset, build_depth_dataset
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import torchvision
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

class DepthEstimator():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(self.device)

    def predict(self, images):
        inputs = self.image_processor(
            images=images,
            do_rescale=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = -torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size = images[0].shape[1:],
            mode="bicubic",
            align_corners=False,
        )

        torch.cuda.empty_cache()
        return prediction

class RgbToRgbd():
    de = DepthEstimator()
    def __call__(self, images):
        return torch.cat((images, self.de.predict(images)), dim=1)

def sample():
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    dataset_name = 'nyu_type3'
    dataset = build_depth_dataset(
        images_dir = cfg['datasets'][dataset_name]['img'],
        gt_dir = cfg['datasets'][dataset_name]['gt'],
        depth_dir = cfg['datasets'][dataset_name]['depth']
    )

    distorted, undistorted = dataset[0]
    images = [undistorted, distorted[:3]]
    gt = distorted[-1]

    de = DepthEstimator()
    prediction = de.predict(images).cpu()

    print(images[0].shape, images[1].shape, prediction.shape)
    
    fig, axs = plt.subplots(ncols=3, nrows=2)
    axs[0, 0].imshow(images[0].permute(1,2,0))
    axs[1, 0].imshow(images[1].permute(1,2,0))
    axs[0, 0].set_title("original images")

    axs[0, 1].imshow(prediction[0, 0])
    axs[1, 1].imshow(prediction[1, 0])
    axs[0, 1].set_title("predicted depths")

    axs[0, 2].imshow(gt)
    axs[0, 2].set_title("true depth")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

if __name__ == "__main__":
    sample()