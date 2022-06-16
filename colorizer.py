import torch

from torch import nn


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()

        self.__l_center = 50
        self.__l_norm = 100

        self.__setup_conv_layers()
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1,
                                   bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_output = self.conv1(self.__normalize_l_input(input_l))
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv2(conv2_output)
        conv4_output = self.conv2(conv3_output)
        conv5_output = self.conv2(conv4_output)
        conv6_output = self.conv2(conv5_output)
        conv7_output = self.conv2(conv6_output)
        conv8_output = self.conv2(conv7_output)

        output = self.model_out(self.softmax(conv8_output))
        output_upscaled = self.upsample4(output)

        return self.__expand_l_input(output_upscaled)

    def __normalize_l_input(self, input):
        return (input - self.__l_center) / self.__l_norm

    def __expand_l_input(self, input):
        return (input * self.__l_norm) + self.__l_center

    def __setup_conv_layers(self):
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        ])

        self.conv3 = nn.Sequential(*[
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        ])

        self.conv4 = nn.Sequential(*[
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ])

        self.conv5 = nn.Sequential(*[
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ])

        self.conv6 = nn.Sequential(*[
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ])

        self.conv7 = nn.Sequential(*[
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ])

        self.conv8 = nn.Sequential(*[
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=313, kernel_size=3, bias=True)
        ])