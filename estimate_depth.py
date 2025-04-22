from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import tomllib
from datasets import build_uied_dataset, build_nyu_dataset
import matplotlib.pyplot as plt
import torch

class DepthEstimator():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(self.device)

    def predict(images):
        raise NotImplementedError


if __name__ == "__main__":
    with open('./config.toml', 'rb') as f:
        cfg = tomllib.load(f)

    dataset_root = cfg['nyu_dataset_root']
    dataset = build_nyu_dataset(
        images_dir=dataset_root+"/type3_data/underwater_type_3",
        depth_maps_dir=dataset_root+"/type3_data/transmission_type_3",
        labels_dir=dataset_root+"/type3_data/gt_type_type_3"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)

    distorted, undistorted = dataset[0]
    print(distorted.shape, undistorted.shape)
    images = [undistorted, distorted[:3]]
    # images = [distorted[:3]]
    gt = distorted[-1]

    inputs = image_processor(
        images=images,
        do_rescale=False,
        return_tensors="pt"
    ).to(device)

    print(inputs['pixel_values'].shape)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    print(images[0].shape)
    print(predicted_depth.shape)

    # interpolate to original size
    prediction = -torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=inputs['pixel_values'].shape[2:],
        mode="bicubic",
        align_corners=False,
    ).cpu()
    
    print(prediction.shape)
    
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