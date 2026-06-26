import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1x1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, 1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv1x1(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SPP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)
        self.S1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1x1(x)
        y1 = self.S1(x)
        y2 = self.S2(x)
        y3 = self.S3(x)
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.relu(x + self.output(y))
        return y


class SPPF(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPPF, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)
        # 定义一个池化层
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.output = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1x1(x)
        y1 = self.relu(self.pool(x))
        y2 = self.relu(self.pool(y1))
        y3 = self.relu(self.pool(y2))
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.relu(x + self.output(y))
        return y


class ConvBlock(nn.Module):
    """
    定义卷积块
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FPNDS(nn.Module):
    def __init__(self, in_channels):
        super(FPNDS, self).__init__()
        C = 128
        # 1x1 卷积用于调整通道数
        self.lateral_p3 = nn.Conv2d(in_channels[2], C, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels[1], C, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(in_channels[0], C, kernel_size=1)
        # 3x3 卷积用于进一步处理融合后的特征
        self.output_p3 = DepthwiseSeparableConv(C, C, 3, 1, 1)
        self.output_p2 = DepthwiseSeparableConv(C, C, 3, 1, 1)
        self.output_p1 = DepthwiseSeparableConv(C, C, 3, 1, 1)

    def forward(self, p1, p2, p3):
        p3_lateral = self.lateral_p3(p3)
        p2_lateral = self.lateral_p2(p2)
        p1_lateral = self.lateral_p1(p1)
        p2_fused = p2_lateral + F.interpolate(
            p3_lateral, size=p2_lateral.shape[2:], mode="nearest"
        )
        p1_fused = p1_lateral + F.interpolate(
            p2_fused, size=p1_lateral.shape[2:], mode="nearest"
        )
        p3_output = self.output_p3(p3_lateral)
        p2_output = self.output_p2(p2_fused)
        p1_output = self.output_p1(p1_fused)
        return p1_output, p2_output, p3_output


class FPNDW(nn.Module):
    def __init__(self, in_channels):
        super(FPNDW, self).__init__()
        C = 128
        # 1x1 卷积用于调整通道数
        self.lateral_p3 = nn.Conv2d(in_channels[2], C, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels[1], C, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(in_channels[0], C, kernel_size=1)
        # 3x3 卷积用于进一步处理融合后的特征
        self.output_p3 = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False)
        self.output_p2 = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False)
        self.output_p1 = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False)

    def forward(self, p1, p2, p3):
        p3_lateral = self.lateral_p3(p3)
        p2_lateral = self.lateral_p2(p2)
        p1_lateral = self.lateral_p1(p1)
        p2_fused = p2_lateral + F.interpolate(
            p3_lateral, size=p2_lateral.shape[2:], mode="nearest"
        )
        p1_fused = p1_lateral + F.interpolate(
            p2_fused, size=p1_lateral.shape[2:], mode="nearest"
        )
        p3_output = self.output_p3(p3_lateral)
        p2_output = self.output_p2(p2_fused)
        p1_output = self.output_p1(p1_fused)
        return p1_output, p2_output, p3_output


class PANDS(nn.Module):
    def __init__(self, in_channels):
        super(PANDS, self).__init__()
        C = 128
        self.lateral_p3 = nn.Conv2d(in_channels[2], C, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels[1], C, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(in_channels[0], C, kernel_size=1)
        self.output_p3 = DepthwiseSeparableConv(C, C, 3, 1, 1)
        self.output_p2 = DepthwiseSeparableConv(C, C, 3, 1, 1)
        self.output_p1 = DepthwiseSeparableConv(C, C, 3, 1, 1)
        self.downsample_p1_to_p2 = DepthwiseSeparableConv(C, C, 3, 2, 1)
        self.downsample_p2_to_p3 = DepthwiseSeparableConv(C, C, 3, 2, 1)

    def forward(self, p1, p2, p3):
        p3_lateral = self.lateral_p3(p3)
        p2_lateral = self.lateral_p2(p2)
        p1_lateral = self.lateral_p1(p1)
        p2_fused = p2_lateral + F.interpolate(
            p3_lateral, size=p2_lateral.shape[2:], mode="nearest"
        )
        p1_fused = p1_lateral + F.interpolate(
            p2_fused, size=p1_lateral.shape[2:], mode="nearest"
        )
        p2_fused_down = p2_fused + self.downsample_p1_to_p2(p1_fused)
        p3_fused_down = p3_lateral + self.downsample_p2_to_p3(p2_fused_down)
        p3_output = self.output_p3(p3_fused_down)
        p2_output = self.output_p2(p2_fused_down)
        p1_output = self.output_p1(p1_fused)
        return p1_output, p2_output, p3_output


class PANDW(nn.Module):
    def __init__(self, in_channels):
        super(PANDW, self).__init__()
        C = 128
        self.lateral_p3 = nn.Conv2d(in_channels[2], C, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels[1], C, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(in_channels[0], C, kernel_size=1)
        self.output_p3 = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False)
        self.output_p2 = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False)
        self.output_p1 = nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False)
        self.downsample_p1_to_p2 = DepthwiseSeparableConv(C, C, 3, 2, 1)
        self.downsample_p2_to_p3 = DepthwiseSeparableConv(C, C, 3, 2, 1)

    def forward(self, p1, p2, p3):
        p3_lateral = self.lateral_p3(p3)
        p2_lateral = self.lateral_p2(p2)
        p1_lateral = self.lateral_p1(p1)
        p2_fused = p2_lateral + F.interpolate(
            p3_lateral, size=p2_lateral.shape[2:], mode="nearest"
        )
        p1_fused = p1_lateral + F.interpolate(
            p2_fused, size=p1_lateral.shape[2:], mode="nearest"
        )
        p2_fused_down = p2_fused + self.downsample_p1_to_p2(p1_fused)
        p3_fused_down = p3_lateral + self.downsample_p2_to_p3(p2_fused_down)
        p3_output = self.output_p3(p3_fused_down)
        p2_output = self.output_p2(p2_fused_down)
        p1_output = self.output_p1(p1_fused)
        return p1_output, p2_output, p3_output


class SPP357(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP357, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)
        self.S5_1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S5_2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S5_3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.S9_1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                1,
                1,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S9_2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                1,
                1,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                1,
                1,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S9_3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                1,
                1,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                1,
                1,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                1,
                1,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S13_1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                7,
                1,
                3,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S13_2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                7,
                1,
                3,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                7,
                1,
                3,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S13_3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                7,
                1,
                3,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                7,
                1,
                3,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                7,
                1,
                3,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1x1(x)
        y5s = [self.S5_1(x), self.S5_2(x), self.S5_3(x)]
        y9s = [self.S9_1(x), self.S9_2(x), self.S9_3(x)]
        y13s = [self.S13_1(x), self.S13_2(x), self.S13_3(x)]
        y5 = torch.cat(y5s, dim=1)
        y5 = self.relu(x + self.output(y5))
        y9 = torch.cat(y9s, dim=1)
        y9 = self.relu(x + self.output(y9))
        y13 = torch.cat(y13s, dim=1)
        y13 = self.relu(x + self.output(y13))
        y = torch.cat((y5, y9, y13), dim=1)
        y = self.relu(x + self.output(y))
        return y


class SPP5913(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP5913, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)
        self.S5_1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S5_2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S5_3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                5,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S9_1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                9,
                1,
                4,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S9_2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                9,
                1,
                4,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                9,
                1,
                4,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S9_3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                9,
                1,
                4,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                9,
                1,
                4,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                9,
                1,
                4,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S13_1 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                13,
                1,
                6,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S13_2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                13,
                1,
                6,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                13,
                1,
                6,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.S13_3 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                13,
                1,
                6,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                13,
                1,
                6,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                13,
                1,
                6,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1x1(x)
        y5s = [self.S5_1(x), self.S5_2(x), self.S5_3(x)]
        y9s = [self.S9_1(x), self.S9_2(x), self.S9_3(x)]
        y13s = [self.S13_1(x), self.S13_2(x), self.S13_3(x)]
        y5 = torch.cat(y5s, dim=1)
        y5 = self.relu(x + self.output(y5))
        y9 = torch.cat(y9s, dim=1)
        y9 = self.relu(x + self.output(y9))
        y13 = torch.cat(y13s, dim=1)
        y13 = self.relu(x + self.output(y13))
        y = torch.cat((y5, y9, y13), dim=1)
        y = self.relu(x + self.output(y))
        return y


class DWConvblock(nn.Module):
    def __init__(self, input_channels, output_channels, size):
        super(DWConvblock, self).__init__()
        self.size = size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                size,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(
                output_channels,
                output_channels,
                size,
                1,
                2,
                groups=output_channels,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LightFPN(nn.Module):
    def __init__(self, input2_depth, input3_depth, out_depth):
        super(LightFPN, self).__init__()
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(input2_depth, out_depth, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_depth),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(input3_depth, out_depth, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_depth),
            nn.ReLU(inplace=True),
        )
        self.cls_head_2 = DWConvblock(input2_depth, out_depth, 5)
        self.reg_head_2 = DWConvblock(input2_depth, out_depth, 5)
        self.reg_head_3 = DWConvblock(input3_depth, out_depth, 5)
        self.cls_head_3 = DWConvblock(input3_depth, out_depth, 5)

    def forward(self, C2, C3):
        S3 = self.conv1x1_3(C3)
        cls_3 = self.cls_head_3(S3)
        obj_3 = cls_3
        reg_3 = self.reg_head_3(S3)
        P2 = F.interpolate(C3, scale_factor=2)
        P2 = torch.cat((P2, C2), 1)
        S2 = self.conv1x1_2(P2)
        cls_2 = self.cls_head_2(S2)
        obj_2 = cls_2
        reg_2 = self.reg_head_2(S2)
        return cls_2, obj_2, reg_2, cls_3, obj_3, reg_3


class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                5,
                1,
                2,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channels, output_channels, 1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        return self.conv5x5(x)


class DetectHead(nn.Module):
    def __init__(self, input_channels, category_num):
        super(DetectHead, self).__init__()
        self.conv1x1 = Conv1x1(input_channels, input_channels)
        self.obj_layers = Head(input_channels, 1)
        self.reg_layers = Head(input_channels, 4)
        self.cls_layers = Head(input_channels, category_num)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1x1(x)
        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))
        return torch.cat((obj, reg, cls), dim=1)


class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


def autopad(k, p=None, d=1):
    """
    kernel, padding, dilation对输入的特征层进行自动padding，按照Same原则
    """
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    """
    标准卷积+标准化+激活函数
    """

    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(
            c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True
        )
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    标准瓶颈结构，残差结构
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSPNet结构结构，大残差结构
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    SPP结构，5、9、13最大池化核的最大池化
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
