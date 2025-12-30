""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, in_kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=in_kernel_size, padding=in_kernel_size//2, bias=True,padding_mode='reflect'),
        # nn.GroupNorm(1,mid_channels),
        nn.LeakyReLU(inplace=True),
        # nn.BatchNorm2d(mid_channels),
        # nn.LeakyReLU(0.1,inplace=True),
        
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
        # nn.GroupNorm(1,out_channels),
        nn.LeakyReLU(inplace=True),
        # nn.BatchNorm2d(out_channels),
        # nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, in_kernel_size=11))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UDenoiseNet(nn.Module):
    # U-net from noise2noise paper (modified stem)
    def __init__(self, nin=1, nf=48, base_width=11, top_width=3):
        super().__init__()
        # 2x(3x3) stem instead of 11x11
        self.enc1 = nn.Sequential(
            nn.Conv2d(nin, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
        )
        # decoder unchanged
        self.dec5 = nn.Sequential(
            nn.Conv2d(2*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(3*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(3*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(3*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*nf, 2*nf, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(2*nf+nin, 64, top_width, padding=top_width//2, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, top_width, padding=top_width//2, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, top_width, padding=top_width//2, padding_mode='reflect'),
        )
    def forward(self, x):
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h  = self.enc6(p5)
        h = F.interpolate(h, size=p4.shape[-2:], mode='nearest')
        h = torch.cat([h, p4], 1)
        h = self.dec5(h)
        h = F.interpolate(h, size=p3.shape[-2:], mode='nearest')
        h = torch.cat([h, p3], 1)
        h = self.dec4(h)
        h = F.interpolate(h, size=p2.shape[-2:], mode='nearest')
        h = torch.cat([h, p2], 1)
        h = self.dec3(h)
        h = F.interpolate(h, size=p1.shape[-2:], mode='nearest')
        h = torch.cat([h, p1], 1)
        h = self.dec2(h)
        h = F.interpolate(h, size=x.shape[-2:], mode='nearest')
        h = torch.cat([h, x], 1)
        return self.dec1(h)
    
class AEUDenoiseNet(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nin=1, nf=48, base_width=11, top_width=3):
        super(AEUDenoiseNet, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(nin, nf, base_width, padding=base_width//2, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf+1, 8, 3, padding=1, padding_mode='reflect')
            , nn.Tanh()
        )
        self.bottleneck2 = nn.Conv2d(8, nf, 3, padding=1, padding_mode='reflect')
                                 

        self.dec5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(nf, nf, 3, padding=1, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(nf, 64, top_width, padding=top_width//2, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2, padding_mode='reflect')
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2, padding_mode='reflect')
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)
        h = torch.cat([h,F.avg_pool2d(x,32)],1)
        h = self.bottleneck2(self.bottleneck(h))

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        y = self.dec1(h)

        return y

class UDenoiseNetRelu(nn.Module):
    # U-net from noise2noise paper
    def __init__(self, nin=1, nf=48, base_width=11, top_width=3):
        super(UDenoiseNetRelu, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(nin, nf, base_width, padding=base_width//2)
                                 , nn.ReLU()
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.ReLU()
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.ReLU()
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+nin, 64, top_width, padding=top_width//2)
                                 , nn.ReLU()
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.ReLU()
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y


class PrepNet(nn.Module):
    def __init__(self, nf=8, width=11):
        super(PrepNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, nf, width, padding=width//2)
            , nn.LeakyReLU(0.1)
            , nn.Conv2d(nf, nf, 3, padding=1)
            , nn.LeakyReLU(0.1)
            , nn.AvgPool2d(2)
            , nn.Conv2d(nf, nf, 3, padding=1)
            , nn.LeakyReLU(0.1)
            , nn.AvgPool2d(2)
            , nn.Conv2d(nf, nf, 3, padding=1)
            , nn.LeakyReLU(0.1)
            , nn.Upsample(scale_factor=2, mode='bilinear')
            , nn.Conv2d(nf, nf, 3, padding=1)
            , nn.LeakyReLU(0.1)
            , nn.Upsample(scale_factor=2, mode='bilinear')
            , nn.Conv2d(nf, nf, 3, padding=1)
            , nn.LeakyReLU(0.1)
            , nn.Conv2d(nf, 1, 3, padding=1)
            
        )
    
    def forward(self, x):
        tmp = self.layers(x)
        print(x.mean(),x.std(),tmp.mean(),tmp.std())
        return tmp


class UDenoiseNetSmall(nn.Module):
    def __init__(self, nf=48, width=11, top_width=3):
        super(UDenoiseNetSmall, self).__init__()

        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, width, padding=width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec3 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        h = self.enc4(p3)

        # upsampling with skip connections
        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y
