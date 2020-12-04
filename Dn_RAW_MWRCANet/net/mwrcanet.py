# -*- coding: utf-8 -*-
# Yue Cao (cscaoyue@gmail.com) (cscaoyue@hit.edu.cn)
# supervisor : Wangmeng Zuo (cswmzuo@gmail.com)
# github: https://github.com/happycaoyue
# personal link:   happycaoyue.com
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
class HITVPCTeam:
    r"""
        DWT and IDWT block written by: Yue Cao
        """
    class CALayer(nn.Module):
        def __init__(self, channel=64, reduction=16):
            super(HITVPCTeam.CALayer, self).__init__()

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv_du(y)
            return x * y

    # conv - prelu - conv - sum
    class RB(nn.Module):
        def __init__(self, filters):
            super(HITVPCTeam.RB, self).__init__()
            self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.act = nn.PReLU()
            self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.cuca = HITVPCTeam.CALayer(channel=filters)

        def forward(self, x):
            c0 = x
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            out = self.cuca(x)
            return out + c0

    class NRB(nn.Module):
        def __init__(self, n, f):
            super(HITVPCTeam.NRB, self).__init__()
            nets = []
            for i in range(n):
                nets.append(HITVPCTeam.RB(f))
            self.body = nn.Sequential(*nets)
            self.tail = nn.Conv2d(f, f, 3, 1, 1)

        def forward(self, x):
            return x + self.tail(self.body(x))

    class DWTForward(nn.Module):
        def __init__(self):
            super(HITVPCTeam.DWTForward, self).__init__()
            ll = np.array([[0.5, 0.5], [0.5, 0.5]])
            lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
            hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
            hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
            filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                              hl[None,::-1,::-1], hh[None,::-1,::-1]],
                             axis=0)
            self.weight = nn.Parameter(
                torch.tensor(filts).to(torch.get_default_dtype()),
                requires_grad=False)
        def forward(self, x):
            C = x.shape[1]
            filters = torch.cat([self.weight,] * C, dim=0)
            y = F.conv2d(x, filters, groups=C, stride=2)
            return y

    class DWTInverse(nn.Module):
        def __init__(self):
            super(HITVPCTeam.DWTInverse, self).__init__()
            ll = np.array([[0.5, 0.5], [0.5, 0.5]])
            lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
            hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
            hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
            filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                              hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                             axis=0)
            self.weight = nn.Parameter(
                torch.tensor(filts).to(torch.get_default_dtype()),
                requires_grad=False)

        def forward(self, x):
            C = int(x.shape[1] / 4)
            filters = torch.cat([self.weight, ] * C, dim=0)
            y = F.conv_transpose2d(x, filters, groups=C, stride=2)
            return y


class Net(nn.Module):
    def __init__(self, channels=1, filters_level1=96, filters_level2=256//2, filters_level3=256//2, n_rb=4*5):
        super(Net, self).__init__()

        self.head = HITVPCTeam.DWTForward()

        self.down1 = nn.Sequential(
            nn.Conv2d(channels * 4, filters_level1, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.NRB(n_rb, filters_level1))

        # sum 1
        # self.down1 = HITVPCTeam.NRB(n_rb, filters_level1),

        # sum 2
        self.down2 = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level1 * 4, filters_level2, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.NRB(n_rb, filters_level2))

        self.down3 = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level2 * 4, filters_level3, 3, 1, 1),
            nn.PReLU())

        self.middle = HITVPCTeam.NRB(n_rb, filters_level3)

        self.up1 = nn.Sequential(
            nn.Conv2d(filters_level3, filters_level2 * 4, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.DWTInverse())

        self.up2 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level2),
            nn.Conv2d(filters_level2, filters_level1 * 4, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.DWTInverse())

        self.up3 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level1),
            nn.Conv2d(filters_level1, channels * 4, 3, 1, 1))

        self.tail = HITVPCTeam.DWTInverse()

    def forward(self, inputs):
        c0 = inputs
        c1 = self.head(c0)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        return self.tail(c7)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
