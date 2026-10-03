import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p=0.0, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.conv1x1 = None
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.conv1x1 is not None:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128, 256, 512), dropout=(0.05, 0.1, 0.2, 0.3, 0.5)):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, channels[0], dropout[0])
        self.down1 = DownBlock(channels[0], channels[1], dropout[1])
        self.down2 = DownBlock(channels[1], channels[2], dropout[2])
        self.down3 = DownBlock(channels[2], channels[3], dropout[3])
        self.down4 = DownBlock(channels[3], channels[4], dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, num_classes, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.up1 = UpBlock(channels[4], channels[3], channels[3], bilinear=True)
        self.up2 = UpBlock(channels[3], channels[2], channels[2], bilinear=True)
        self.up3 = UpBlock(channels[2], channels[1], channels[1], bilinear=True)
        self.up4 = UpBlock(channels[1], channels[0], channels[0], bilinear=True)
        self.out_conv = nn.Conv2d(channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features
        x = self.up1(x4, x3)
        out3 = x  # 256 channels
        x = self.up2(x, x2)
        out2 = x  # 128 channels
        x = self.up3(x, x1)
        out1 = x  # 64 channels
        x = self.up4(x, x0)
        logits = self.out_conv(x)
        return logits, out1, out2, out3


class UNet(nn.Module):
    """U-Net backbone used by PUMix.

    Decoder features returned as (D1, D2, D3) have 64, 128 and 256 channels,
    matching the multi-scale prototype construction in the original notebook.

    The decoder explicitly uses bilinear upsampling + 1x1 channel projection,
    matching the actual UpBlock behavior in the research notebook.
    """

    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.encoder(x)
        return self.decoder(features)
