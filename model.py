import torch.nn as nn
import torch.nn.parallel
import torch
import pdb

dd = pdb.set_trace

v_siz = 9
z_siz = 128 - v_siz


class conv_mean_pool(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(conv_mean_pool, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.pooling(out)
        return out


class mean_pool_conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(mean_pool_conv, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = x
        out = self.pooling(out)
        out = self.conv(out)
        return out

class upsample_conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(upsample_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(inplanes, outplanes, 3, 1, 1)

    def forward(self, x):
        out = x
        out = self.upsample(out)
        out = self.conv(out)
        return out


class residualBlock_down(nn.Module): # for discriminator, no batchnorm
    def __init__(self, inplanes, outplanes):
        super(residualBlock_down, self).__init__()
        self.conv_shortcut = mean_pool_conv(inplanes, outplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
        self.conv2 = conv_mean_pool(outplanes, outplanes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


class residualBlock_up(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(residualBlock_up, self).__init__()
        self.conv_shortcut = upsample_conv(inplanes, outplanes)
        self.conv1 = upsample_conv(inplanes, outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = x
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.ReLU(out)
        out = self.conv2(out)

        return shortcut + out


class _G_xvz(nn.Module):
    def __init__(self):
        super(_G_xvz, self).__init__()
        #self.conv = nn.Conv2d(3, 64, 3, 1, 1) 64*64 resolution implementation
        self.conv = nn.Conv2d(3, 64, 3, 1, 1) # 3*128*128 --> 64*128*128
        self.resBlock0 = residualBlock_down(64, 64) # 64*128*128 --> 64*64*64
        self.resBlock1 = residualBlock_down(64, 128)
        self.resBlock2 = residualBlock_down(128, 256)
        self.resBlock3 = residualBlock_down(256, 512)
        self.resBlock4 = residualBlock_down(512, 512)
        self.fc_v = nn.Linear(512*4*4, v_siz)
        self.fc_z = nn.Linear(512*4*4, z_siz)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.resBlock0(out)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = out.view(-1, 512*4*4)
        v = self.fc_v(out)
        v = self.softmax(v)
        z = self.fc_z(out)

        return v, z


class _G_vzx(nn.Module):
    def __init__(self):
        super(_G_vzx, self).__init__()
        # inplanes, outplanes, kernel_size, stride, padding
        # H_out = (H_in-1)*stride[0] - 2*padding[0] + kernel_size[0] + output_padding[0]
        # W_out = (W_in-1)*stride[1] - 2*padding[1] + kernel_size[1] + output_padding[1]
        self.fc = nn.Linear(v_siz+z_siz, 4*4*512)
        self.resBlock1 = residualBlock_up(512, 512) #4*4-->8*8
        self.resBlock2 = residualBlock_up(512, 256) #8*8-->16*16
        self.resBlock3 = residualBlock_up(256, 128) #16*16-->32*32
        self.resBlock4 = residualBlock_up(128, 64) #32*32-->64*64
        self.resBlock5 = residualBlock_up(64, 64) #64*64-->128*128
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, v, z):
        x = torch.cat((v,z), 1)
        out = self.fc(x) # out: 512*4*4
        out = out.view(-1, 512, 4, 4) # (-1, 512, 4, 4)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.resBlock3(out)
        out = self.resBlock4(out)
        out = self.resBlock5(out)
        out = self.bn(out)
        out = self.ReLU(out)
        out = self.conv(out)
        out = self.tanh(out)

        return out


class _D_xvs(nn.Module):
    def __init__(self):
        super(_D_xvs, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1) #3*64*64 --> 64*64*64
        #self.conv = nn.Conv2d(3, 64, 7, 2, 3) #3*128*128 --> 64*64*64
        self.resBlock0 = residualBlock_down(64, 64)
        self.resBlock1 = residualBlock_down(64, 128) #64*64*64 --> 119*32*32
        self.resBlock2 = residualBlock_down(128, 256) #128*32*32 --> 256*16*16
        self.resBlock3 = residualBlock_down(256, 512) #256*16*16 --> 512*8*8
        self.resBlock4 = residualBlock_down(512, 512) #512*8*8 --> 512*4*4
        self.fc_v = nn.Linear(512*4*4, v_siz)
        self.fc_s = nn.Linear(512*4*4, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.resBlock0(x)
        x = self.resBlock1(x) # 119*32*32
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = x.view(-1, 512*4*4)
        v = self.fc_v(x)
        v = self.softmax(v)
        s = self.fc_s(x)

        return v, s
