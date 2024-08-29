# yuukilight
# yuukilight
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn

# 建立模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(1, 10, 7, padding='same'),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.MaxPool1d(4),  # 8192->2048
            nn.Conv1d(10, 20, 7, padding='same'),
            nn.BatchNorm1d(20),
            nn.Tanh(),
            nn.MaxPool1d(4),  # 2048->512
            nn.Conv1d(20, 10, 7, padding='same'),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.MaxPool1d(2),  # 512->256
            nn.Flatten(),
            nn.Linear(2560, 64),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(64, 5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.module(x)
        return x


# DRSN

# channels width height c w h
class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        # BRC = BatchNormation + ReKU + Convolution
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地修改参数
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        # GAP
        # AdaptiveAvgPool1d 会自动调节 kernel and stride
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        # Flatten 需要输入展开的维度范围，默认从 1 到所有维度。（只保留 0 维度，即 batch）
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        # sign(x) 用于返回符号 -1，0，1
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding = torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result


class DRSNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=4, padding=2),
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, stride=4, padding=2)
        )
        self.bn = nn.BatchNorm1d(40)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 = nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)
        self.linear = nn.Linear(in_features=40, out_features=20)
        self.output_class = nn.Linear(in_features=20, out_features=1)
        self.RSBU_CW = nn.Sequential(
            RSBU_CW(in_channels=10, out_channels=10, kernel_size=3, down_sample=True),
            RSBU_CW(in_channels=10, out_channels=10, kernel_size=3, down_sample=False),
            RSBU_CW(in_channels=10, out_channels=20, kernel_size=3, down_sample=True),
            RSBU_CW(in_channels=20, out_channels=20, kernel_size=3, down_sample=False),
            RSBU_CW(in_channels=20, out_channels=40, kernel_size=3, down_sample=True),
            RSBU_CW(in_channels=40, out_channels=40, kernel_size=3, down_sample=False)
        )

    def forward(self, input):  #
        x = self.conv1(input)  #
        x = self.RSBU_CW(x)  #
        x = self.bn(x)  # 40*64
        x = self.relu(x)
        gap = self.global_average_pool(x)  # 40*1
        gap = self.flatten(gap)  # 1*40
        linear1 = self.linear(gap)  # 1*20
        output_class = self.output_class(linear1)  # 1*3
        #         output_class = self.softmax(output_class)  # 1*3
        return output_class

class RSU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        # BRC = BatchNormation + ReLU + Convolution
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地修改参数
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        # GAP
        # AdaptiveAvgPool1d 会自动调节 kernel and stride
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        # Flatten 需要输入展开的维度范围，默认从 1 到所有维度。（只保留 0 维度，即 batch）
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding = torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result



class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=4, padding=2),
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, stride=4, padding=2)
        )
        self.bn = nn.BatchNorm1d(40)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 = nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)
        self.linear = nn.Linear(in_features=40, out_features=20)
        self.output_class = nn.Linear(in_features=20, out_features=1)
        self.RSBU_CW = nn.Sequential(
            RSU(in_channels=10, out_channels=10, kernel_size=3, down_sample=True),
            RSU(in_channels=10, out_channels=10, kernel_size=3, down_sample=False),
            RSU(in_channels=10, out_channels=20, kernel_size=3, down_sample=True),
            RSU(in_channels=20, out_channels=20, kernel_size=3, down_sample=False),
            RSU(in_channels=20, out_channels=40, kernel_size=3, down_sample=True),
            RSU(in_channels=40, out_channels=40, kernel_size=3, down_sample=False)
        )

    def forward(self, input):  #
        x = self.conv1(input)  #
        x = self.RSBU_CW(x)  #
        x = self.bn(x)  # 40*64
        x = self.relu(x)
        gap = self.global_average_pool(x)  # 40*1
        gap = self.flatten(gap)  # 1*40
        linear1 = self.linear(gap)  # 1*20
        output_class = self.output_class(linear1)  # 1*3
        #         output_class = self.softmax(output_class)  # 1*3
        return output_class


class DCU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        # BRC = BatchNormation + ReLU + Convolution
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地修改参数
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        # GAP
        # AdaptiveAvgPool1d 会自动调节 kernel and stride
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        # Flatten 需要输入展开的维度范围，默认从 1 到所有维度。（只保留 0 维度，即 batch）
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding = torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return x

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=4, padding=2),
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, stride=4, padding=2)
        )
        self.bn = nn.BatchNorm1d(40)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear6_8 = nn.Linear(in_features=256, out_features=128)
        self.linear8_4 = nn.Linear(in_features=128, out_features=64)
        self.linear4_2 = nn.Linear(in_features=64, out_features=32)
        self.output_center_pos = nn.Linear(in_features=32, out_features=1)
        self.output_width = nn.Linear(in_features=32, out_features=1)
        self.linear = nn.Linear(in_features=40, out_features=20)
        self.output_class = nn.Linear(in_features=20, out_features=1)
        self.RSBU_CW = nn.Sequential(
            DCU(in_channels=10, out_channels=10, kernel_size=3, down_sample=True),
            DCU(in_channels=10, out_channels=10, kernel_size=3, down_sample=False),
            DCU(in_channels=10, out_channels=20, kernel_size=3, down_sample=True),
            DCU(in_channels=20, out_channels=20, kernel_size=3, down_sample=False),
            DCU(in_channels=20, out_channels=40, kernel_size=3, down_sample=True),
            DCU(in_channels=40, out_channels=40, kernel_size=3, down_sample=False)
        )

    def forward(self, input):  #
        x = self.conv1(input)  #
        x = self.RSBU_CW(x)  #
        x = self.bn(x)  # 40*64
        x = self.relu(x)
        gap = self.global_average_pool(x)  # 40*1
        gap = self.flatten(gap)  # 1*40
        linear1 = self.linear(gap)  # 1*20
        output_class = self.output_class(linear1)  # 1*3
        #         output_class = self.softmax(output_class)  # 1*3
        return output_class


if __name__ == '__main__':
    # model = CNN()
    # model.cuda()
    # model.double()
    # x = torch.rand(10, 1, 8192)
    # x = x.cuda()
    # x = x.double()
    # out = model(x)
    # print(out.size())
    print("main branch")
    print("tmp_branch")
    print("has merge")
