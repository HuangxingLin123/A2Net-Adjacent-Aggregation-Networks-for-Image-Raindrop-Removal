import skimage
import cv2
from skimage.measure import compare_psnr, compare_ssim



def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    return compare_ssim(im1_y, im2_y)


## ground truth
dir='./datasets/test/clean/'


##  derained results
dir4="./results/pretrained/test_latest/images/"


ssim=0
psnr=0

total_num=177
for i in range(177):

    img1= cv2.imread(dir+str(i)+'_clean.jpg')
    (h, w, n) = img1.shape
    img2 = cv2.imread(dir4 + str(i) +'_rain_result.png')

    (h2, w2, n) = img2.shape
    if h2 != h or w2 != w:
        print('error', i)
        break

    a = calc_ssim(img1, img2)
    b = calc_psnr(img1, img2)
    print(i, ':', a, b)
    ssim += a
    psnr += b




ssim=ssim/total_num
psnr=psnr/total_num
print("ssim=",ssim)
print("psnr=",psnr)




