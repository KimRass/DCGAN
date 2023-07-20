import torch
import torch.nn as nn
import torch.nn.functional as F


class FractionallyStridedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activ, batchnorm=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activ = activ
        self.batchnorm = batchnorm

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
        )
        # "Third is Batch Normalization. Directly applying batchnorm to all layers however, resulted in sample
        # oscillation and model instability. This was avoided by not applying batchnorm to the generator
        # output layer and the discriminator input layer."
        if batchnorm:
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.norm(x)
        # "The ReLU activation is used in the generator with the exception of the output layer which uses
        # the Tanh function. We observed that using a bounded activation allowed the model to learn
        # more quickly to saturate and cover the color space of the training distribution."
        # for higher resolution"
        if self.activ == "relu":
            x = torch.relu(x)
        elif self.activ == "tanh":
            x = torch.tanh(x)
        return x


def _init_weights(model):
    for m in model.modules():
        # "All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02."
        if isinstance(m, (nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(0, 0.02)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # "A 100 dimensional uniform distribution `Z` is projected to a small spatial extent convolutional representation
        # with many feature maps."
        self.proj = nn.Linear(100, 1024 * 4 * 4)
        # "A series of four fractionally-strided convolutions then convert this high level representation
        # into a $64 \times 64$ pixel image. No fully connected or pooling layers are used."

        self.block1 = FractionallyStridedConvBlock(1024, 512, activ="relu")
        self.block2 = FractionallyStridedConvBlock(512, 256, activ="relu")
        self.block3 = FractionallyStridedConvBlock(256, 128, activ="relu")
        self.block4 = FractionallyStridedConvBlock(128, 3, activ="tanh", batchnorm=False)

        _init_weights(self)

    def forward(self, x): # `(b, 100)`
        # "The first layer of the GAN, which takes a uniform noise distribution `Z` as input, could be called
        # fully connected as it is just a matrix multiplication, but the result is reshaped into
        # a 4-dimensional tensor and used as the start of the convolution stack."
        x = self.proj(x) # `(b, 1024 * 4 * 4)`
        x = F.relu(x)
        x = x.view(-1, 1024, 4, 4) # `(b, 1024, 4, 4)`

        x = self.block1(x) # `(b, 1024, 4, 4)`
        x = self.block2(x) # `(b, 512, 8, 8)`
        x = self.block3(x) # `(b, 256, 16, 16)`
        x = self.block4(x) # `(b, 128, 32, 32)`
        return x # `(b, 3, 64, 64)`


class StridedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False)
        if batchnorm:
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.norm(x)
        # "Use LeakyReLU activation in the discriminator for all layers. The slope of the leak
        # was set to 0.2 in all models."
        x = F.leaky_relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = StridedConvBlock(3, 128, batchnorm=False)
        self.block2 = StridedConvBlock(128, 256)
        self.block3 = StridedConvBlock(256, 512)
        self.block4 = StridedConvBlock(512, 1024)

        self.proj = nn.Linear(1024 * 4 * 4, 1)

        _init_weights(self)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # "The last convolution layer is flattened and then fed into a single sigmoid output."
        x = x.view(-1, 1024 * 4 * 4)
        x = self.proj(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 100)
    gen = Generator()
    gen(x).shape

    x = torch.randn(2, 3, 64, 64)
    disc = Discriminator()
    disc(x).shape

