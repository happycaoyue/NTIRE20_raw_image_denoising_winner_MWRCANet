# -*- coding: utf-8 -*-
# Yue Cao (cscaoyue@gmail.com) (cscaoyue@hit.edu.cn)
# supervisor : Wangmeng Zuo (cswmzuo@gmail.com)
# github: https://github.com/happycaoyue
# personal link:   happycaoyue.com
import os
import random
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import cv2
from net.mwrcanet import Net
random.seed()
import math
def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def main():
    print("********************Experiment on SIDDPLUS Raw Space valid dataset********************")
    # pretrained model dir
    last_ckpt = './net_last_ckpt/dn_mwrcanet_raw_c1.pth'

    # net architecture
    dn_net = Net()

    # Move to GPU
    dn_model = nn.DataParallel(dn_net).cuda()

    # load old params
    tmp_ckpt = torch.load(last_ckpt)
    # Initialize dn_model
    pretrained_dict = tmp_ckpt['state_dict']
    model_dict=dn_model.state_dict()
    pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    assert(len(pretrained_dict)==len(pretrained_dict_update))
    assert(len(pretrained_dict_update)==len(model_dict))
    model_dict.update(pretrained_dict_update)
    dn_model.load_state_dict(model_dict)

    # load valid dataset
    mat_dir = './mat/siddplus_valid_noisy_raw.mat'
    # Read .mat file
    mat_file = sio.loadmat(mat_dir)
    # 1024 × 256 × 256
    noisyblock = mat_file['siddplus_valid_noisy_raw']
    print('input shape', noisyblock.shape)
    resultblock = noisyblock

    mat_dir = './mat/siddplus_valid_gt_raw.mat'
    # Read .mat file
    mat_file = sio.loadmat(mat_dir)
    # 1024 × 256 × 256
    denoisedblock = mat_file['siddplus_valid_gt_raw']
    print('denoised shape', denoisedblock.shape)



    # test
    dn_model.eval()
    with torch.no_grad():
        psnr_val = 0
        # psnr_val2 = 0
        count = 0

        for kk in range(1024):
            # 256 × 256
            img = np.array(noisyblock[kk], dtype=np.float32)
            label = np.array(denoisedblock[kk], dtype=np.float32)
            GT = label
            origin = img
            count += 1
            print("ValImg_ID: %d" % (count))
            # (1，256, 256)
            origin = np.expand_dims(origin, 0)
            # 256 * 256 * 1
            origin = np.transpose(origin, (1, 2, 0))
            noisy = origin.copy()

            noisy_data = []
            out_data = []
            out_data_real = []

            out_ = np.zeros(origin.shape)
            output = np.zeros(origin.shape)
            noisy_ = np.zeros((origin.shape[2], origin.shape[0], origin.shape[1]))
            temp1 = np.zeros((origin.shape[2], origin.shape[1], origin.shape[0]))
            temp2 = np.zeros((origin.shape[2], origin.shape[0], origin.shape[1]))
            temp3 = np.zeros((origin.shape[2], origin.shape[1], origin.shape[0]))

            # (1, 256, 256)
            for a in range(1):
                noisy_[a, :, :] = noisy[:, :, a]

            # rotate / flip
            noisy_data.append(noisy_)
            for a in range(1):
                temp1[a, :, :] = np.rot90(noisy_[a, :, :], 1)
                temp2[a, :, :] = np.rot90(noisy_[a, :, :], 2)
                temp3[a, :, :] = np.rot90(noisy_[a, :, :], 3)
            noisy_data.append(temp1)
            noisy_data.append(temp2)
            noisy_data.append(temp3)

            noisy_data.append(np.fliplr(noisy_data[0]).copy())
            noisy_data.append(np.fliplr(noisy_data[1]).copy())
            noisy_data.append(np.fliplr(noisy_data[2]).copy())
            noisy_data.append(np.fliplr(noisy_data[3]).copy())

            for x in range(8):
                img = np.expand_dims(noisy_data[x], 0)
                input = torch.tensor(img).cuda().float()
                out = dn_model(input)
                out_data.append(out.cpu().data[0].numpy().astype(np.float32))
                # if x == 0:
                #     out_no = out.cpu().data[0].numpy().astype(np.float32)

            for a in range(8):
                out_data_real.append(np.zeros((origin.shape[2], origin.shape[0], origin.shape[1])))

            out_data[4] = np.fliplr(out_data[4])
            out_data[5] = np.fliplr(out_data[5])
            out_data[6] = np.fliplr(out_data[6])
            out_data[7] = np.fliplr(out_data[7])

            for a in range(1):
                out_data_real[1][a, :, :] = np.rot90(out_data[1][a, :, :], -1)
                out_data_real[2][a, :, :] = np.rot90(out_data[2][a, :, :], -2)
                out_data_real[3][a, :, :] = np.rot90(out_data[3][a, :, :], -3)

                out_data_real[5][a, :, :] = np.rot90(out_data[5][a, :, :], -1)
                out_data_real[6][a, :, :] = np.rot90(out_data[6][a, :, :], -2)
                out_data_real[7][a, :, :] = np.rot90(out_data[7][a, :, :], -3)

            out_data_real[0] = out_data[0]
            out_data_real[4] = out_data[4]

            for x in range(8):
                for a in range(1):
                    out_[:, :, a] = out_data_real[x][a, :, :]
                output += out_
            output /= 8.0
            output[output < 0] = 0
            output[output > 1] = 1.0

            # out_no[out_no < 0] = 0
            # out_no[out_no > 1] = 1.0
            # out_no = out_no.astype(np.float32)
            # out_no = out_no[0]

            output = output.astype(np.float32)
            output = output.transpose(2, 0, 1)
            output = np.squeeze(output)
            psnr = output_psnr_mse(output, GT)

            # psnr2 = output_psnr_mse(out_no, GT)

            psnr_val += psnr
            # psnr_val2 += psnr2

            print(psnr)


        psnr_val /= 1024.0
        # psnr_val2 /= 1024.0
        print("Initial model avg val PSNR:")
        print(psnr_val)

        # print("Initial model avg val PSNR:")
        # print(psnr_val2)



if __name__ == "__main__":
    main()
    exit(0)



