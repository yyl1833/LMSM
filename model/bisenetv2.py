import torch
from torch import nn

__all__ = ['BiseNetV2']


class ConvBNLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dialation=1, groups=1, bias=False):
        super(ConvBNLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, dilation=dialation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNLU(3, 64, 3, stride=2),
            ConvBNLU(64, 64, 3, stride=1)
        )
        self.S2 = nn.Sequential(
                    ConvBNLU(64, 64, 3, stride=2),
                    ConvBNLU(64, 64, 3, stride=1),
                    ConvBNLU(64, 64, 3, stride=1)
                )
        self.S3 = nn.Sequential(
            ConvBNLU(64, 128, 3, stride=2),
            ConvBNLU(128, 128, 3, stride=1),
            ConvBNLU(128, 128, 3, stride=1)
        )

    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x

class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNLU(16, 8, 1, stride=1, padding=0),
            ConvBNLU(8, 16, 3, stride=2, padding=1)
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNLU(32, 16, 3, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x_left = self.left(x)
        x_right = self.right(x)
        # 此处是变宽(dim=1)
        x = torch.concat([x_left, x_right], dim=1)
        x = self.fuse(x)
        return x


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNLU(128, 128, 3, stride=1)

    def forward(self, x):
        x1 = torch.mean(x, dim=(2, 3), keepdim=True)   # dim()表示消除size()对应位置的数(N, C, H, W)
        x1 = self.bn(x1)
        x1 = self.conv_gap(x1)
        x = x1 + x
        x = self.conv_last(x)
        return x


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNLU(in_chan, mid_chan, 3, stride=1)
        # 此处和源码chan不一样
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan)
        )

        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.dwconv(x1)
        x1 = self.conv2(x1)
        x = x + x1
        x = self.relu(x)
        return x


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNLU(in_chan, mid_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, 3, stride=2, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan)
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan)
        )

        self.conv2[1].last_bn = True

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(in_chan, out_chan, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.dwconv1(x1)
        x1 = self.dwconv2(x1)
        x1 = self.conv2(x1)
        x2 = self.shortcut(x)
        x = x1 + x2
        x = self.relu(x)
        return x


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32, 6),
            GELayerS1(32, 32, 6)
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64, 6),
            GELayerS1(64, 64, 6)
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128, 6),
            GELayerS1(128, 128, 6),
            GELayerS1(128, 128, 6),
            GELayerS1(128, 128, 6)
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        x2 = self.S1S2(x)
        x3 = self.S3(x2)
        x4 = self.S4(x3)
        x5_4 = self.S5_4(x4)
        x5_5 = self.S5_5(x5_4)
        return x2, x3, x4, x5_4, x5_5


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False)
        )
        self.left_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )
        self.right_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

    def forward(self, x_d, x_s):
        left_1 = self.left_1(x_d)
        left_2 = self.left_2(x_d)
        right_1 = self.right_1(x_s)
        right_2 = self.right_2(x_s)
        x_left = left_1 * right_1
        x_right = left_2 * right_2
        x_right = self.up(x_right)
        x = x_left + x_right
        x = self.conv(x)
        return x


# 此处后续需要优化
class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_class, up_factor=8):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNLU(in_chan, mid_chan, 3, stride=1, padding=1)
        self.drop = nn.Dropout(0.1)

        out_chan = n_class
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, 1, stride=1, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.conv_out(x)
        return x


class BiseNetV2(nn.Module):

    def __init__(self, n_class, mode='train'):
        super(BiseNetV2, self).__init__()
        self.mode = mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        # mid_chan不知道是不是1024
        self.head = SegmentHead(128, 64, n_class, up_factor=8)
        if self.mode == 'train':
            self.aux2 = SegmentHead(16, 128, n_class, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_class, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_class, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_class, up_factor=32)

        # 这个不知道为什么要加,训练策略?
        self.init_weight()

    def forward(self, x):
        x_d = self.detail(x)
        x2, x3, x4, x5_4, x_s = self.segment(x)
        x = self.bga(x_d, x_s)

        logit = self.head(x)
        if self.mode == 'train':
            logit_aux2 = self.aux2(x2)
            logit_aux3 = self.aux3(x3)
            logit_aux4 = self.aux4(x4)
            logit_aux5_4 = self.aux5_4(x5_4)
            return logit, logit_aux2, logit_aux3, logit_aux4, logit_aux5_4
        elif self.mode == 'eval':
            return logit
        else:
            raise NotImplementedError

    def init_weight(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)



# if __name__ == '__main__':
    # x = torch.randn(2, 3, 1024, 2048)
    # net = DetailBranch()
    # x = net(x)
    # print(x.shape)

    # x = torch.randn(2, 3, 1024, 2048)
    # net = StemBlock()
    # x = net(x)
    # print(x.shape)

    # x = torch.randn(2, 128, 32, 64)
    # net = CEBlock()
    # x = net(x)
    # print(x.shape)

    # x = torch.randn(2, 8, 32, 64)
    # net = GELayerS1(8, 8)
    # x = net(x)
    # print(x.shape)

    # x = torch.randn(2, 8, 32, 64)
    # net = GELayerS2(8, 32)
    # x = net(x)
    # print(x.shape)

    # x = torch.randn(2, 128, 128, 256)
    # y = torch.randn(2, 128, 32, 64)
    # net = BGALayer()
    # z = net(x, y)
    # print(z.shape)

    # x = torch.randn(2, 32, 128, 256)
    # net = SegmentHead(32, 64, 19, 8)
    # x = net(x)
    # print(x.shape)

    # x = torch.randn(2, 3, 1024, 2048)
    # net = SegmentBranch()
    # x = net(x)[4]
    # print(x.shape)

    # x = torch.randn(5, 3, 736, 1280)
    # net = BiseNetV2(n_class=19)
    # net = BiseNetV2(n_class=19, mode='eval')
    # net.eval()
    # outs = net(x)
    # for out in outs:
    #     print(out.shape)
    # print(net)


































