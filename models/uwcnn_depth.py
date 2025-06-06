import torch
import torch.nn as nn
import torch.nn.functional as F

class UWCNN_Depth(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_layers()

    def _init_layers(self):
        self.conv2d_cc0 = nn.Conv2d(4, 4, 1, 1)
        self.cc0_relu = nn.ReLU(inplace=True)
        self.conv2d_cc1 = nn.Conv2d(4, 4, 1, 1)
        self.cc1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze1 = nn.Conv2d(12, 16, 3, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(4+16+16+16, 16, 3, 1, 1)
        self.dehaze4_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze5_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze6_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze7 = nn.Conv2d(52+48, 16, 3, 1, 1)
        self.dehaze7_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze8_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze9_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze10 = nn.Conv2d(100+48, 3, 3, 1, 1)

    def forward(self, x):
        image_cc0 = self.cc0_relu(self.conv2d_cc0(x))
        image_cc1 = self.cc1_relu(self.conv2d_cc1(image_cc0))
        cc_concat = torch.cat([image_cc0, image_cc1, x], dim=1)

        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(cc_concat))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, x], dim=1)
        image_conv4 = self.dehaze4_relu(self.conv2d_dehaze4(dehaze_concat1))
        image_conv5 = self.dehaze5_relu(self.conv2d_dehaze5(image_conv4))
        image_conv6 = self.dehaze6_relu(self.conv2d_dehaze6(image_conv5))

        dehaze_concat2 = torch.cat([dehaze_concat1, image_conv4, image_conv5, image_conv6], dim=1)
        image_conv7 = self.dehaze7_relu(self.conv2d_dehaze7(dehaze_concat2))
        image_conv8 = self.dehaze8_relu(self.conv2d_dehaze8(image_conv7))
        image_conv9 = self.dehaze9_relu(self.conv2d_dehaze9(image_conv8))

        dehaze_concat3 = torch.cat([dehaze_concat2, image_conv7, image_conv8, image_conv9], dim=1)
        image_conv10 = self.conv2d_dehaze10(dehaze_concat3)
        out = x[:, :3] + image_conv10

        return out
    
class Small_UWCNN_Depth(nn.Module):
    '''
    Final E block removed
    '''
    def __init__(self):
        super().__init__()
        self._init_layers()

    def _init_layers(self):
        self.conv2d_cc0 = nn.Conv2d(4, 4, 1, 1)
        self.cc0_relu = nn.ReLU(inplace=True)
        self.conv2d_cc1 = nn.Conv2d(4, 4, 1, 1)
        self.cc1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze1 = nn.Conv2d(12, 16, 3, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(4+16+16+16, 16, 3, 1, 1)
        self.dehaze4_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze5_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze6_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze7 = nn.Conv2d(52+48, 3, 3, 1, 1)

    def forward(self, x):
        image_cc0 = self.cc0_relu(self.conv2d_cc0(x))
        image_cc1 = self.cc1_relu(self.conv2d_cc1(image_cc0))
        cc_concat = torch.cat([image_cc0, image_cc1, x], dim=1)

        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(cc_concat))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, x], dim=1)
        image_conv4 = self.dehaze4_relu(self.conv2d_dehaze4(dehaze_concat1))
        image_conv5 = self.dehaze5_relu(self.conv2d_dehaze5(image_conv4))
        image_conv6 = self.dehaze6_relu(self.conv2d_dehaze6(image_conv5))

        dehaze_concat2 = torch.cat([dehaze_concat1, image_conv4, image_conv5, image_conv6], dim=1)
        image_conv7 = self.conv2d_dehaze7(dehaze_concat2)
        out = x[:, :3] + image_conv7

        return out
    
class Tiny_UWCNN_Depth(nn.Module):
    '''
    Last two E block removed
    '''
    def __init__(self):
        super().__init__()
        self._init_layers()

    def _init_layers(self):
        self.conv2d_cc0 = nn.Conv2d(4, 4, 1, 1)
        self.cc0_relu = nn.ReLU(inplace=True)
        self.conv2d_cc1 = nn.Conv2d(4, 4, 1, 1)
        self.cc1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze1 = nn.Conv2d(12, 16, 3, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(52, 3, 3, 1, 1)

    def forward(self, x):
        image_cc0 = self.cc0_relu(self.conv2d_cc0(x))
        image_cc1 = self.cc1_relu(self.conv2d_cc1(image_cc0))
        cc_concat = torch.cat([image_cc0, image_cc1, x], dim=1)

        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(cc_concat))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, x], dim=1)
        image_conv4 = self.conv2d_dehaze4(dehaze_concat1)
        out = x[:, :3] + image_conv4

        return out
    
class Micro_UWCNN_Depth(nn.Module):
    '''
    Last two E block removed
    '''
    def __init__(self):
        super().__init__()
        self._init_layers()

    def _init_layers(self):
        self.conv2d_dehaze1 = nn.Conv2d(4, 8, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(8, 8, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(8, 8, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(28, 3, 1, 1)

    def forward(self, x):
        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(x))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, x], dim=1)
        image_conv4 = self.conv2d_dehaze4(dehaze_concat1)
        out = x[:, :3] + image_conv4

        return out

class UWCNN_DepthPrepass(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_layers()

    def _init_layers(self):
        self.linear_pp1 = nn.Conv2d(4, 4, 1)
        self.pp1_relu = nn.ReLU(inplace=True)

        self.linear_pp2 = nn.Conv2d(4, 3, 1)
        self.pp2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(3+16+16+16, 16, 3, 1, 1)
        self.dehaze4_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze5_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze6_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze7 = nn.Conv2d(51+48, 16, 3, 1, 1)
        self.dehaze7_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze8_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze9_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze10 = nn.Conv2d(99+48, 3, 3, 1, 1)

    def forward(self, x):
        im_pp1 = self.pp1_relu(self.linear_pp1(x))
        im_pp2 = self.pp2_relu(self.linear_pp2(im_pp1))

        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(im_pp2))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, im_pp2], dim=1)
        image_conv4 = self.dehaze4_relu(self.conv2d_dehaze4(dehaze_concat1))
        image_conv5 = self.dehaze5_relu(self.conv2d_dehaze5(image_conv4))
        image_conv6 = self.dehaze6_relu(self.conv2d_dehaze6(image_conv5))

        dehaze_concat2 = torch.cat([dehaze_concat1, image_conv4, image_conv5, image_conv6], dim=1)
        image_conv7 = self.dehaze7_relu(self.conv2d_dehaze7(dehaze_concat2))
        image_conv8 = self.dehaze8_relu(self.conv2d_dehaze8(image_conv7))
        image_conv9 = self.dehaze9_relu(self.conv2d_dehaze9(image_conv8))

        dehaze_concat3 = torch.cat([dehaze_concat2, image_conv7, image_conv8, image_conv9], dim=1)
        image_conv10 = self.conv2d_dehaze10(dehaze_concat3)
        out = x[:, :3] + image_conv10

        return out

if __name__ == "__main__":
    from torchinfo import summary
    uwcnn_d = Micro_UWCNN_Depth()
    summary(uwcnn_d, (1, 4, 224, 224))