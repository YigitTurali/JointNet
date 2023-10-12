from PIL import Image
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import math
import math
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
# Example usage:
        #python3 /auto/data2/odalmaz/psnr_ssim_pgan.py --fake_dir /auto/data2/odalmaz/TransResNet_residual_configs/config_r0_5/results/T1_T2_IXI_ViT_config_r0_5_both_pre_trained_sgd_0.001/test_9/images/ --normalize 1 --IXI 1
#
def calculate_psnr_ssim(fake_dir,IXI=True,normalize=True):
    print('Calculating PSNR and SSIM for validation set')
    if normalize:
        print("Normalized")
    else:
        print("Not normalized")
    if IXI:
        n_slices = 2165
    else:
        n_slices = 1999
    psnr_vals = []
    ssim_vals = []
    for slice_ind in range(n_slices):
        if (90 <= slice_ind+1 and slice_ind+1 <= 100)or (200 < slice_ind+1 and slice_ind+1 <= 205) or (305 < slice_ind+1 and slice_ind+1 <= 420) or (533 < slice_ind+1 and slice_ind+1 <= 540) or (1606 <= slice_ind+1 and slice_ind+1 <= 1715 and IXI):
             continue
        real_image = Image.open(fake_dir + str(slice_ind+1) + '_real_B.png').convert("L")
        vit_fake_image = Image.open(fake_dir + str(slice_ind+1) + '_fake_B.png').convert("L")
        real_image = np.asarray(real_image, dtype='float64')
        vit_fake_image = np.asarray(vit_fake_image, dtype='float64')

        if not normalize:
            psnr_vals.append(peak_signal_noise_ratio(real_image, fake_image, data_range=255))
            ssim_vals.append(structural_similarity(real_image, fake_image, data_range=255))
        else:
            if np.max(real_image) == 0:
                continue
                print(np.max(real_image))
            real_image /= np.max(real_image)
            vit_fake_image /= np.max(vit_fake_image)
            psnr_vit = peak_signal_noise_ratio(real_image, vit_fake_image, data_range=1)
            psnr_vals.append(psnr_vit)
            ssim_vals.append(compare_ssim(real_image, vit_fake_image, data_range=1))

    psnr_vals = np.array(psnr_vals)
    ssim_vals = np.array(ssim_vals)
    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)

    std_psnr = np.std(psnr_vals)
    std_ssim = np.std(ssim_vals)

    return mean_psnr,std_psnr,mean_ssim,std_ssim


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_dir', type=str, required=True,
                       help='Path to the fake image folder')
    parser.add_argument('--normalize', type=int, default=1,
                        help='Path to the fake image folder')
    parser.add_argument('--IXI', type=int, default=1,
                        help='Path to the fake image folder')
    opt = parser.parse_args()

    #

    fake_dir = opt.fake_dir
    print(fake_dir)
    mean_psnr,std_psnr,mean_ssim,std_ssim=calculate_psnr_ssim(fake_dir,opt.IXI,opt.normalize)

    print("PSNR:")
    print("MEAN :" + str(mean_psnr))
    print("STD :" + str(std_psnr))

    print("SSIM:")
    print("MEAN :" + str(mean_ssim))
    print("STD :" + str(std_ssim))

    print('Done')


#
# ############################### difference calculator
# from PIL import Image
# import numpy as np
# from skimage.measure import compare_psnr, compare_ssim
# from skimage.metrics import peak_signal_noise_ratio,structural_similarity
# import math
# import math
# import numpy as np
# import cv2
# from skimage.io import imread
# from skimage.color import rgb2gray
# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))
#
# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
# # Example usage:
# #python3 /auto/data2/odalmaz/psnr_ssim_pgan.py --real_dir /auto/data2/odalmaz/TransResNet_residual_configs/config_r0/results/T1_T2__PD_IXI_resnet_dropout_no_init/test_latest/images/  --fake_dir /auto/data2/odalmaz/TransResNet_residual_configs/config_r0/results/T1_T2__PD_IXI_resnet_dropout_no_init/test_latest/images/ --normalize 1 --IXI 1
# #
# def calculate_psnr_ssim(real_dir,fake_dir,IXI=True,normalize=True):
#     print('Calculating PSNR and SSIM for validation set')
#     if normalize:
#         print("Normalized")
#     else:
#         print("Not normalized")
#     if IXI:
#         n_slices = 2165
#     else:
#         n_slices = 1999
#     psnr_vals = []
#     ssim_vals = []
#     for slice_ind in range(n_slices):
# #        if 0 <= slice_ind and slice_ind <= 9:
# #            no = '00' + str(slice_ind)
# #        elif 10 <= slice_ind and slice_ind <= 99:
# #            no = '0' + str(slice_ind)
# #        else:
# #            no = str(slice_ind)
# #        # print(real_dir+ no+".png")
#         if (90 <= slice_ind+1 and slice_ind+1 <= 100)or (200 < slice_ind+1 and slice_ind+1 <= 205) or (305 < slice_ind+1 and slice_ind+1 <= 420) or (533 < slice_ind+1 and slice_ind+1 <= 540) or (1606 <= slice_ind+1 and slice_ind+1 <= 1715):
#              continue
#         real_image = Image.open(real_dir + str(slice_ind+1) + '_real_B.png').convert("L") #Image.open(real_dir + no + ".png").convert("L")(result_dir,num2str(slice_ind),'_real_B.png')
#         resnet_fake_image = Image.open(real_dir + str(slice_ind+1) + '_fake_B.png').convert("L") #Image.open(real_dir + no + ".png").convert("L")
#         vit_fake_image = Image.open(fake_dir + str(slice_ind+1) + '_fake_B.png').convert("L")
#         real_image = np.asarray(real_image, dtype='float64')
#         resnet_fake_image = np.asarray(resnet_fake_image, dtype='float64')
#         vit_fake_image = np.asarray(vit_fake_image, dtype='float64')
#
#         if not normalize:
#             psnr_vals.append(peak_signal_noise_ratio(real_image, fake_image, data_range=255))
#             ssim_vals.append(structural_similarity(real_image, fake_image, data_range=255))
#         else:
#             if np.max(real_image) == 0:
#                 continue
#                 print(np.max(real_image))
#             real_image /= np.max(real_image)
#             resnet_fake_image /= np.max(resnet_fake_image)
#             vit_fake_image /= np.max(vit_fake_image)
#             psnr_resnet = peak_signal_noise_ratio(real_image, resnet_fake_image, data_range=1)
#             psnr_vit = peak_signal_noise_ratio(real_image, vit_fake_image, data_range=1)
#             if psnr_vit - psnr_resnet > 2:
#                 print(slice_ind+1," :", psnr_vit - psnr_resnet, " dB" )
#             psnr_vals.append(psnr_vit)
#             ssim_vals.append(compare_ssim(real_image, vit_fake_image, data_range=1))
#
#     # mean_psnr = 0
#     # mean_ssim = 0
#     # no_test_instances = 0
#     # for i in range(2165):
#     #     # we should ignore test samples between [305,420] and [1604,1715]
#     #     if (305 <= i and i <= 420) or (1604 <= i and i <= 1715):
#     #         continue
#     #     mean_psnr = mean_psnr + psnr_vals[i]
#     #     mean_ssim = mean_ssim + ssim_vals[i]
#     #     no_test_instances = no_test_instances + 1
#     # std_psnr = 0
#     # std_ssim = 0
#     #
#     # for i in range(2165):
#     #     # we should ignore test samples between [305,420] and [1604,1715]
#     #     if (305 <= i and i <= 420) or (1604 <= i and i <= 1715):
#     #         continue
#     #     std_psnr = std_psnr + (psnr_vals[i] - mean_psnr) ** 2
#     #     std_ssim = std_ssim + (ssim_vals[i] - mean_ssim) ** 2
#     psnr_vals = np.array(psnr_vals)
#     ssim_vals = np.array(ssim_vals)
#     mean_psnr = np.mean(psnr_vals)
#     mean_ssim = np.mean(ssim_vals)
#     #
#     # std_psnr = np.std()
#     # std_ssim = std_ssim / no_test_instances
#
#     std_psnr = np.std(psnr_vals)
#     std_ssim = np.std(ssim_vals)
#
#     return mean_psnr,std_psnr,mean_ssim,std_ssim
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--real_dir', type=str, required=True,
#                         help='Path to the real image folder')
#     parser.add_argument('--fake_dir', type=str, required=True,
#                        help='Path to the fake image folder')
#     parser.add_argument('--normalize', type=int, default=1,
#                         help='Path to the fake image folder')
#     parser.add_argument('--IXI', type=int, default=1,
#                         help='Path to the fake image folder')
#     opt = parser.parse_args()
#
#     #
#
#     print(opt.real_dir)
#     real_dir = opt.real_dir
#     fake_dir = opt.fake_dir
#
#     mean_psnr,std_psnr,mean_ssim,std_ssim=calculate_psnr_ssim(real_dir, fake_dir,opt.IXI,opt.normalize)
#
#     print("PSNR:")
#     print("MEAN :" + str(mean_psnr))
#     print("STD :" + str(std_psnr))
#
#     print("SSIM:")
#     print("MEAN :" + str(mean_ssim))
#     print("STD :" + str(std_ssim))
#
#     print('Done')
