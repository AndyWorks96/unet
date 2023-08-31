import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from utils import str2bool, count_params


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class Unet01(nn.Module):
    def __init__(self, args):
        in_chan = 4
        out_chan = 3
        N=64
        n1=N
        n2=2*N
        n3=4*N
        n4=8*N
        n5=16*N
        super(Unet01, self).__init__()
        self.down1 = Downsample_block(in_chan, n1)
        self.down2 = Downsample_block(n1, n2)
        self.down3 = Downsample_block(n2, n3)
        self.down4 = Downsample_block(n3, n4)
        self.conv1 = nn.Conv2d(n4, n5, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n5)
        self.conv2 = nn.Conv2d(n5, n5, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n5)
        self.up4 = Upsample_block(n5, n4)
        self.up3 = Upsample_block(n4, n3)
        self.up2 = Upsample_block(n3, n2)
        self.up1 = Upsample_block(n2, n1)
        self.outconv = nn.Conv2d(n1, out_chan, 1)
        self.outconvp1 = nn.Conv2d(n1, out_chan, 1)
        self.outconvm1 = nn.Conv2d(n1, out_chan, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        # b = torch.nn.functional.adaptive_max_pool2d(x,(1,1))
        # bb = torch.nn.functional.adaptive_max_pool1d(x,1)
        feature01 = torch.nn.functional.adaptive_max_pool2d(x,(1,1))
        x, y2 = self.down2(x)
        feature02 = torch.nn.functional.adaptive_max_pool2d(x,(1,1))
        x, y3 = self.down3(x)
        feature03 = torch.nn.functional.adaptive_max_pool2d(x,(1,1))
        x, y4 = self.down4(x)
        feature04 = torch.nn.functional.adaptive_max_pool2d(x, (1, 1))

        # features = feature01 + feature02 + feature03 + feature04
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)

        return x1


if __name__ == '__main__':

    model = Unet01(nn.Module)
    args = None
    # create model
    device = 'cpu'
    models = model.to(device)
    # solution: 1
    model = models.cpu()
    summary(model,(4,160,160))
    # print(model)
    # print(count_params(model))

