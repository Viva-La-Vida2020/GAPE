import torch
import torch.nn as nn
import torch.nn.functional as F
from nowcasting.layers.utils import spectral_norm


# Temporal_Discriminator
class Temporal_Discriminator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_length = configs.input_length
        # self.total_length = configs.total_length
        self.total_length = configs.total_length

        self.img_height = configs.img_height
        self.img_width = configs.img_width
        self.conv2d = torch.nn.Conv2d(in_channels=self.total_length, out_channels=64, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4))
        self.conv3d_1 = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(self.input_length, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4))
        self.conv3d_2 = torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(self.total_length-self.input_length, 9, 9), stride=(1, 2, 2), padding=(0, 4, 4))
        self.down_1 = Down(228, 128)
        self.down_2 = Down(128, 256)
        self.down_3 = Down(256, 512)
        self.l_block = LBlock(512, 512)
        self.head = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(2e-1),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=3, padding=3 // 2)),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Sigmoid()
        # )

    def forward(self, x):  # x: torch.Size([16, 24, 512, 512])
        x = torch.cat((self.conv2d(x),  # torch.Size([16, 64, 256, 256])
                       torch.flatten(self.conv3d_1(x.unsqueeze(1)), start_dim=1, end_dim=2),  # torch.Size([16, 4, 21, 256, 256])
                       torch.flatten(self.conv3d_2(x.unsqueeze(1)), start_dim=1, end_dim=2)), dim=1)  # torch.Size([16, 8, 5, 256, 256])
        # x1 = self.conv2d(x)
        # x2 = self.conv3d_1(x.unsqueeze(1))
        # x3 = self.conv3d_2(x.unsqueeze(1))
        #  x: torch.Size([16, 188, 512, 512])
        x = self.down_1(x)  # torch.Size([16, 128, 128, 128])
        x = self.down_2(x)  # torch.Size([16, 256, 64, 64])
        x = self.down_3(x)  # torch.Size([16, 512, 32, 32])
        x = self.l_block(x) # torch.Size([16, 512, 32, 32])
        x = self.head(x)    # torch.Size([16, 1, 32, 32])
        x = self.avg_pool(x) # [B, 1, 1, 1]
        'loss使用BCEWithLogitsLoss因此不需要sigmoid'
        x = x.view(x.size(0), -1)  # [B, 1]
        # x = self.classifier(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            LBlock(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class LBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        # if not mid_channels:
        #     mid_channels = out_channels
        self.relu_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2)),
        )
        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2))
        )

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.relu_conv(x)
        x = x + shortcut
        return x