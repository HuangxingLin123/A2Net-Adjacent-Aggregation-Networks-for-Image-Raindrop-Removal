import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import pytorch_ssim
import numpy as np



class DerainModel(BaseModel):
    def name(self):
        return 'DerainModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser



    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        if self.isTrain:

            self.visual_names = [ 'X_rgb','Y_rgb','result']
        else:
            self.visual_names = ['result']

        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,id=2)
        self.netG = networks.define_G(opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.ssim_loss = pytorch_ssim.SSIM()

            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)


    def set_input(self, input,epoch,iteration):
        if self.isTrain:
            self.X_rgb = input['X_rgb'].to(self.device)
            self.Y_rgb = input['Y_rgb'].to(self.device)
            self.X_yuv = input['X_yuv'].to(self.device)
            self.Y_yuv = input['Y_yuv'].to(self.device)

            self.X_y = input['X_y'].to(self.device)
            self.X_uv = input['X_uv'].to(self.device)

            self.Y_y = input['Y_y'].to(self.device)
            self.Y_uv = input['Y_uv'].to(self.device)

            self.image_paths = input['X_paths']
        else:
            self.X_rgb = input['X_rgb'].to(self.device)
            self.X_yuv = input['X_yuv'].to(self.device)
            self.X_y = input['X_y'].to(self.device)
            self.X_uv = input['X_uv'].to(self.device)
            self.image_paths = input['X_paths']


    def forward(self):
        (self.res_UV, self.fake_Y, self.fake_UV, self.result) = self.netG(self.X_y, self.X_uv)



    def backward_G2(self):
        self.loss_y_SSIM = (1 - self.ssim_loss(self.fake_Y, self.Y_y)) * 1
        self.loss_uv_SSIM = (1 - self.ssim_loss(self.fake_UV, self.Y_uv)) * 0.6


        self.loss_G2 = self.loss_y_SSIM  + self.loss_uv_SSIM

        self.loss_G2.backward()




    def optimize_parameters(self):
        self.forward()
        # update D

        self.optimizer_G.zero_grad()
        self.backward_G2()
        self.optimizer_G.step()