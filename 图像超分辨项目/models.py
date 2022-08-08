import torch
from torch import nn
import torchvision
import math


class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    子像素卷积模块, 包含卷积, 像素清洗和激活层.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :参数 kernel_size: 卷积核大小
        :参数 n_channels: 输入和输出通道数
        :参数 scaling_factor: 放大比例
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # 进行像素清洗，合并相关通道数据
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # 最后添加激活层
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    残差模块, 包含两个卷积模块和一个跳连.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :参数 kernel_size: 核大小
        :参数 n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）
        """
        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(SRResNet, self).__init__()

        # 放大比例必须为 2、 4 或 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs

