import torch
from src.layer import ConvBlock, DownSample, UpSample
from src.model import UNet


def test_conv_block():
    inputs = torch.zeros([1, 3, 572, 572], dtype=torch.float32)

    sut = ConvBlock(3, 64)
    out = sut.forward(inputs)

    assert out.shape == (1, 64, 572, 572)


def test_down_sample():
    inputs = torch.zeros([1, 3, 572, 572], dtype=torch.float32)

    sut = DownSample(3, 64)
    out = sut.forward(inputs)

    assert out.shape == (1, 64, 286, 286)


def test_up_sample():
    inputs = torch.zeros([1, 1024, 28, 28], dtype=torch.float32)
    r_inputs = torch.zeros([1,  512, 64, 64])

    sut = UpSample(1024, 512)
    out = sut.forward(inputs, r_inputs)

    assert out.shape == (1, 512, 64, 64)


def test_unet():
    inputs = torch.zeros([1, 3, 572, 572], dtype=torch.float32)
    n_classes = 10

    sut = UNet(3, n_classes)
    out = sut.forward(inputs)

    assert out.shape == (1, 10, 572, 572)
