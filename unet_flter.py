import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import str2bool, count_params
import numpy as np
from glob import glob

class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
    # 78*78
    def forward(self, x):
        fff=self.bn1(self.conv1(x))
        zz=F.relu(x)
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(y, 2, stride=2)
        x=F.avg_pool2d(y,2,stride=2)
        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        # 反卷积
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1,output_padding=1, stride=2)
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


class Unet(nn.Module):
    def __init__(self, args):
        in_chan = 4
        out_chan = 3
        super(Unet, self).__init__()
        self.down1 = Downsample_block(in_chan, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)
        self.outconvp1 = nn.Conv2d(64, out_chan, 1)
        self.outconvm1 = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        npimage = np.load('./data/testImage1/BraTS20_Training_001_30.npy')
        # npmask = np.load(mask_paths)
        npimage = npimage.transpose((2, 0, 1))
        # print(npimage)
        npimage=npimage.astype(np.float32)
        xx = torch.from_numpy(npimage)  # numpy转换torch
        x = torch.unsqueeze(xx, dim=0)


        x, y1 = self.down1(x)

        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)

        return x1


if __name__ == '__main__':

    model = Unet(nn.Module)
    args = None
    # create model
    device = 'cpu'
    models = model.to(device)
    # solution: 1
    model = models.cpu()
    # Data loading code
    # img_paths = glob('./data/testImage1/BraTS20_Training_001_30.npy')
    # mask_paths = glob('./data/testMask1/BraTS20_Training_001_30.npy')
    npimage = np.load('./data/testImage1/BraTS20_Training_001_30.npy')
    # npmask = np.load(mask_paths)
    npimage = npimage.transpose((2, 0, 1))

    summary(model,(4,160,160))
    # summary(model,npimage)
    # print(model)
    print(count_params(model))