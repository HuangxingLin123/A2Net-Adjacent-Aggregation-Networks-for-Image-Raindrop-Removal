# A2Net-Adjacent-Aggregation-Networks-for-Image-Raindrop-Removal

Requirements
========

Linux

pytorch

python3.6


Training
======

Download the training dataset from ...

Unzip 'train.zip' in './datasets/'. 

Make sure the training images are in the './datasets/train/rain/' and './datasets/train/clean/', respectively.

- Train the deraining model:

*python train.py --dataroot ./datasets/train/rain/ --name new --model derain*



Testing
=======

Download the testing dataset from [Google Drive](https://drive.google.com/drive/folders/1ps9u1KyI_W0c0hJaZ_gKWwG8hJgd7KKA).

Unzip 'test.zip' in './datasets/'.

- Test:

*python test.py --dataroot ./datasets/test/rain/ --name new --model derain*

- Test with our pretrained model:

*python test.py --dataroot ./datasets/test/rain/ --name pretrained --model derain*

After the test, results are saved in './results/'.

Run "psnr_and_ssim.py" to caculate psnr and ssim.
