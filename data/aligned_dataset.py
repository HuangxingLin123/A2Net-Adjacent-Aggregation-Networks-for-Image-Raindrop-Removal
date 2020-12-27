import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import cv2




class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.istrain = opt.isTrain
        self.root = opt.dataroot
        self.dir_X = os.path.join(opt.dataroot)
        self.X_paths = sorted(make_dataset(self.dir_X))



    def __getitem__(self, index):
        if self.istrain:
            X_path = self.X_paths[index]
            X = cv2.imread(X_path)
            (h, w, n) = X.shape

            Y_name = X_path.split('train/rain/')[1].split('rain.')[0]

            # Y_b_name = X_path.split('datasets/train/rainy_image/')[1]
            Y = cv2.imread('./datasets/train/clean/' + Y_name + 'clean.png')

            width = 256

            h_off = random.randint(0, h - width)
            w_off = random.randint(0, w - width)
            X = X[h_off:h_off + width, w_off:w_off + width]
            Y = Y[h_off:h_off + width, w_off:w_off + width]

            rr = random.randint(0, 3)

            if rr == 1:
                X = cv2.flip(X, 0)
                Y = cv2.flip(Y, 0)
            elif rr == 2:
                X = cv2.flip(X, 1)
                Y = cv2.flip(Y, 1)
            elif rr == 3:
                X = cv2.flip(X, -1)
                Y = cv2.flip(Y, -1)
            else:
                pass


            X_rgb = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
            X_yuv = cv2.cvtColor(X, cv2.COLOR_BGR2YUV)

            Y_rgb = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)
            Y_yuv = cv2.cvtColor(Y, cv2.COLOR_BGR2YUV)




            X_rgb = transforms.ToTensor()(X_rgb)
            Y_rgb = transforms.ToTensor()(Y_rgb)
            X_yuv = transforms.ToTensor()(X_yuv)
            Y_yuv = transforms.ToTensor()(Y_yuv)
            X_y =  X_yuv[0, :, :].unsqueeze(0)
            X_uv = X_yuv[1:3, :, :]
            Y_y = Y_yuv[0, :, :].unsqueeze(0)
            Y_uv = Y_yuv[1:3, :, :]


            return {'X_rgb': X_rgb,'Y_rgb': Y_rgb,'X_yuv': X_yuv, 'Y_yuv': Y_yuv, 'X_y': X_y, 'X_uv': X_uv,'Y_y': Y_y, 'Y_uv': Y_uv, 'X_paths': X_path}
        else:

            X_path = self.X_paths[index]
            X = cv2.imread(X_path)
            X_rgb = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
            X_yuv = cv2.cvtColor(X, cv2.COLOR_BGR2YUV)

            X_rgb = transforms.ToTensor()(X_rgb)
            X_yuv = transforms.ToTensor()(X_yuv)
            X_y = X_yuv[0, :, :].unsqueeze(0)
            X_uv = X_yuv[1:3, :, :]

            return {'X_rgb': X_rgb,'X_yuv': X_yuv, 'X_y': X_y, 'X_uv': X_uv, 'X_paths': X_path}



    def __len__(self):
        return len(self.X_paths)

    def name(self):
        return 'AlignedDataset'
