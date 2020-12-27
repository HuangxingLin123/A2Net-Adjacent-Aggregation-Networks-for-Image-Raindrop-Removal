import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, input):
        x=self.conv1(input)
        d1=self.down1(input)
        d2=self.down2(x)
        d0=torch.cat((d1,d2),dim=1)
        a=self.conv2(d0)

        return a,x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True,inner_most=False,outer_most=False):
        super(up, self).__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.up1=nn.Sequential(
            nn.ConvTranspose2d(out_ch , out_ch , 2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, input2,input3):
        input1=self.conv1(input2)
        x1=self.up1(input1)
        x2=self.up2(input2)
        merge=torch.cat((x1,x2,input3),dim=1)
        x=self.conv2(merge)

        return x
class up_UV(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True,inner_most=False,outer_most=False):
        super(up_UV, self).__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.up1=nn.Sequential(
            nn.ConvTranspose2d(out_ch , out_ch , 2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*out_ch+32, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, input2,input3):
        input1=self.conv1(input2)
        x1=self.up1(input1)
        x2=self.up2(input2)
        merge=torch.cat((x1,x2,input3),dim=1)
        x=self.conv2(merge)

        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


