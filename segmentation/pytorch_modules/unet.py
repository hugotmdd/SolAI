import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet101, ResNet101_Weights

decoder_params = {
    "num_channels_in": 2048,
    "concats": [1024, 512, 256, 64],
    "num_classes": 1,
    "dilation": 1,
} 

class Encoder(torch.nn.Module):
    def __init__(self, path=None):
        super(Encoder, self).__init__()
        self.nn = torch.nn.Sequential(
            *list(resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).children())[:8]
        )
        self.layers = [3, 5, 6, 7, 8]

    def forward(self, imput):
        features = [imput]
        for layer in self.nn.children():
            features.append(layer(features[-1]))
        return list(map(lambda x: features[x], self.layers))


class UpSample(nn.Sequential):
    def __init__(self, input_features, output_features, concat_c, norm=True):
        super(UpSample, self).__init__()

        self.norm = norm
        self.convA = nn.Conv2d(
            input_features + concat_c,
            output_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if norm:
            self.gnA = nn.GroupNorm(num_channels=output_features, num_groups=32)
            self.gnB = nn.GroupNorm(num_channels=output_features, num_groups=32)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(
            output_features, output_features, kernel_size=3, stride=1, padding=1
        )
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x, scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        x = self.convA(torch.cat([up_x, concat_with], dim=1))
        if self.norm:
            x = self.gnA(x)
        x = self.convB(self.leakyreluB(x))
        if self.norm:
            x = self.gnB(x)
        x = self.leakyreluB(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_channels_in=1664,
        dilation=1,
        concats=[256, 128, 64, 64],
        num_classes=2,
        mult=1,
        deconv=False,
    ):
        super(Decoder, self).__init__()
        self.deconv = deconv
        self.bridge = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels_in, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        num_channels_in = 128
        self.upblock_1 = UpSample(
            num_channels_in, num_channels_in // (2 * mult), concat_c=concats[0]
        )
        self.upblock_2 = UpSample(
            num_channels_in // (2 * mult),
            num_channels_in // (4 * mult),
            concat_c=concats[1],
        )
        self.upblock_3 = UpSample(
            num_channels_in // (4 * mult),
            num_channels_in // (8 * mult),
            concat_c=concats[2],
            norm=False,
        )
        self.upblock_4 = UpSample(
            num_channels_in // (8 * mult),
            num_channels_in // (16 * mult),
            concat_c=concats[3],
            norm=False,
        )

        self.conv_last = torch.nn.Conv2d(
            num_channels_in // (16 * mult),
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, featput):
        x_block0, x_block1, x_block2, x_block3, x_block4 = featput
        x_d0 = self.bridge(x_block4)
        x_d1 = self.upblock_1(x_d0, x_block3)
        x_d2 = self.upblock_2(x_d1, x_block2)
        x_d3 = self.upblock_3(x_d2, x_block1)
        x_d4 = self.upblock_4(x_d3, x_block0)
        x_d5 = self.conv_last(x_d4)
        x_d6 = torch.nn.functional.interpolate(
            x_d5, scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        return x_d6

class SegModel(torch.nn.Module):
    def __init__(self, encoder_path=None):
        super(SegModel, self).__init__()

        self.name = "unet"
        self.encoder = Encoder(path=encoder_path)
        self.decoder = Decoder(**decoder_params)

    def forward(self, imput):
        features = self.encoder(imput)
        return self.decoder(features)
