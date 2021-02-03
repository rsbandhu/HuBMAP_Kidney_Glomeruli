import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/yassouali/pytorch_segmentation#training
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#pytorch template
#https://github.com/victoresque/pytorch-template/blob/master/README.md



from base.base_model import BaseModel

class Conv2x(nn.Module):
    '''
    preserves the the size of the image
    '''
    def __init__(self, in_ch, out_ch, inner_ch=None):
        super(Conv2x, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inner_ch = out_ch//2 if inner_ch is None else inner_ch

        self.conv2d_1 = nn.Conv2d(self.in_ch, self.inner_ch,
                                  kernel_size=3, padding=1, bias=False)
        self.conv2d_2 = nn.Conv2d(self.inner_ch, self.out_ch,
                                  kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inner_ch)
        self.bn2 = nn.BatchNorm2d(self.out_ch)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class encoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(encoder, self).__init__()
        self.conv2x = Conv2x(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv2x(x)
        x = self.pool(x)
        return x

class decoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.transposeconv = nn.ConvTranspose2d(
            in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv2x = Conv2x(in_ch, out_ch)

    def forward(self, x_down, x_up, interpolate=True):

        x_up = self.transposeconv(x_up)

        #check for matching dims before concatenating

        if (x_up.size(2) != x_up.size(2)) or (x_up.size(3) != x_up.size(3)):
            if interpolate:
                x_up = F.interpolate(x_up, size=(x_down.size(2), x_down.size(3)),
                mode="bilinear", align_corners=True)
        
        #Concat features from down conv channel and current up-conv
        #along channel dim =1
        x_up = torch.cat([x_up, x_down], dim=1) 
        x_up = self.conv2x(x_up)

        return x_up

class UNet(BaseModel):

    def __init__(self, in_ch=3, conv_channels=[16, 32, 64, 128, 256]):
        super(UNet, self).__init__()

        self.conv_channels = conv_channels
        self.conv_start = Conv2x(in_ch, conv_channels[0]) #output_size = input_size
        self.down1 = encoder(conv_channels[0], conv_channels[1])   #output_size = input_size/2
        self.down2 = encoder(conv_channels[1], conv_channels[2])   #output_size = input_size/2
        self.down3 = encoder(conv_channels[2], conv_channels[3])   #output_size = input_size/2
        self.down4 = encoder(conv_channels[3], conv_channels[4])   #output_size = input_size/2

        self.conv_middle = Conv2x(conv_channels[4], conv_channels[4]) #output_size = input_size

        self.up4 = decoder(conv_channels[4], conv_channels[3]) #output_size = input_size * 2
        self.up3 = decoder(conv_channels[3], conv_channels[2]) #output_size = input_size * 2
        self.up2 = decoder(conv_channels[2], conv_channels[1]) #output_size = input_size * 2
        self.up1 = decoder(conv_channels[1], conv_channels[0]) #output_size = input_size * 2

        self.final_conv = nn.Conv2d(self.conv_channels[0], 1, kernel_size=1)

        self.init_params()
    
    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



    def forward(self, x):
        # size of x = [B, _, nx, ny]
        
        x1 = self.conv_start(x)  # size of x = [B, self.conv_channels[0], nx, ny]
        x2 = self.down1(x1)  # size of x = [B, self.conv_channels[1], nx/2, ny/2]
        x3 = self.down2(x2)  # size of x = [B, self.conv_channels[2], nx/4, ny/4]
        x4 = self.down3(x3)  # size of x = [B, self.conv_channels[3], nx/8, ny/8]
        x5 = self.down4(x4)  # size of x = [B, self.conv_channels[4], nx/16, ny/16]

        x = self.conv_middle(x5)  # size of x = [B, self.conv_channels[4], nx/16, ny/16]

        x = self.up4(x4, x)       # size of x = [B, self.conv_channels[3], nx/8, ny/8]
        x = self.up3(x3, x)       # size of x = [B, self.conv_channels[2], nx/4, ny/4]
        x = self.up2(x2, x)       # size of x = [B, self.conv_channels[1], nx/2, ny/2]
        x = self.up1(x1, x)       # size of x = [B, self.conv_channels[0], nx, ny]

        x = self.final_conv(x)

        return x









