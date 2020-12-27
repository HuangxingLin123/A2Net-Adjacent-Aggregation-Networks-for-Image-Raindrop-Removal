import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .unet_parts import *


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = A2Net(c_num=16)


    return init_net(net, init_type, init_gain, gpu_ids)



class A2Net(nn.Module):
    def __init__(self, c_num=16):
        super(A2Net, self).__init__()
        channel_num=c_num
        self.inc = nn.Sequential(
            nn.Conv2d(3, 2 * channel_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down1 = down(2 * channel_num, 2 * channel_num)
        self.down2 = down(2 * channel_num, 2 * channel_num)
        self.down3 = down(2 * channel_num, 2 * channel_num)

        self.up_UV1 = up_UV(2 * channel_num, 24)
        self.up_UV2 = up_UV(24, 24)
        self.up_UV3 = up_UV(24, 24)
        self.outc_UV = nn.Sequential(
            nn.Conv2d(24, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.up1 = up(2 * channel_num, 2 * channel_num)
        self.up2 = up(2 * channel_num, 2 * channel_num)
        self.up3 = up(2 * channel_num, 2 * channel_num)
        self.outc_Y = nn.Sequential(
            nn.Conv2d(2 * channel_num, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, Y,UV):
        input = torch.cat((Y, UV), 1)
        a0 = self.inc(input)
        a1, x1 = self.down1(a0)
        a2, x2 = self.down2(a1)
        a3, x3 = self.down3(a2)

        output_UV = self.up_UV1(a3, x3)
        output_UV = self.up_UV2(output_UV, x2)
        output_UV = self.up_UV3(output_UV, x1)
        res_UV = self.outc_UV(output_UV)
        output_UV=res_UV+UV

        output_Y = self.up1(a3, x3)
        output_Y = self.up2(output_Y, x2)
        output_Y = self.up3(output_Y, x1)
        res_Y = self.outc_Y(output_Y)
        output_Y=res_Y+Y

        output_YUV=torch.cat((output_Y.detach(),output_UV.detach()),1)


        return res_UV,output_Y,output_UV,output_YUV



