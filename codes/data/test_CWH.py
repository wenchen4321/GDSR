import os as os
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
import PIL.Image as Image
from codes.data.util import modcrop
import scipy.misc


#path = '/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR/'
# path = '/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR_new/Flickr2K/'
# #path = '/home/lcc/CWH/Dataset2/Flickr2K_LR_bicubic/2K_train_LRx4_new/'
# files = os.listdir(path)
# i=0
# for img in files:
#       tmp = plt.imread(path+img)
#       i=i+1
#       print(i)
#       print(img)

    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    # if env is None:  # img
    #     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # else:
    #     img = _read_lmdb_img(env, path)

# path="/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR_new/error/HR/002060.png"
# img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# if img is None:
#       print("image:!!!!!" + path)
# img = img.astype(np.float32) / 255.
# print(img)
# # img = img.astype(np.float32) / 255.
# if img.ndim == 2:
#       img = np.expand_dims(img, axis=2)
#     # some images have 4 channels
# if img.shape[2] > 3:
#       img = img[:, :, :3]

#path = "/media/lcc/CWH/LR/HR/ILSVRC2012_val_00019877.JPEG"
path = "/media/lcc/CWH/LR/gray/LR/ILSVRC2012_val_00000760.png"
scale = 4
image = cv2.imread(path)
print(image.shape)
# label_ = modcrop(image, scale)
# im_LR = cv2.imresize(label_, 1/4, 'bicubic')
# print("2")
# cv2.imwrite('/media/lcc/CWH/LR/HR/1.JPEG', im_LR)
# print("3")
# print("end")
# #       img1 = img1.astype(np.float32) / 255.
# # if img.ndim == 2:
# #       img = np.expand_dims(img, axis=2)
# #     # some images have 4 channels
# # if img.shape[2] > 3:
# #       img = img[:, :, :3]
#       i = i + 1
#       print(i)
#       print(img)
# files = os.listdir(path)
# i = 0
# for img in files:
# #       tmp = plt.imread(path+img)
# #       print(img)
# #       img1 = cv2.imread(path+img, cv2.IMREAD_UNCHANGED)
# #       img1 = img1.astype(np.float32) / 255.
# # if img.ndim == 2:
# #       img = np.expand_dims(img, axis=2)
# #     # some images have 4 channels
# # if img.shape[2] > 3:
# #       img = img[:, :, :3]
#       i = i + 1
#       print(i)
#       print(img)

# 将所有的图片转换成为jpg格式（防止因为图片格式造成的cv2.imread()异常）


# filelist = os.listdir(path)
# for file in filelist:
#
#       img = Image.open(path + file).convert('RGB')
#             # print(img)
#       img.save(path + file)
#       print('Done!')


# for filename in files:
#     print(filename)
#images = os.listdir('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR')
#images = os.listdir('./')
# tmp = cv2.imread('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR/000077.png')
# print(tmp.shape)
#images = os.path.join('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR', images)
# for img in files:
#     tmp = plt.imread(path+img)
#     print(tmp.shape)
    # if  tmp.shape[2] == 0:
    #     #img = os.path.join('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR', img)
    #     print(tmp.shape)
    # if tmp.shape[2] == 128:
    #     #img = os.path.join('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR', img)
    #     print(tmp.shape)

# folder_name = '/home/lcc/CWH/Dataset2/Flickr2K_LR_bicubic/2K_LR_800_x4/DIV2K_train_LR_x4'
# file_names = os.listdir(folder_name)
# os.chdir(folder_name)
# #3对获取的名字重命名
# # for name in file_names:
# #       print(name)
# #       os.rename(name,'[mittake出品]-'+name)
#
# for name in file_names:
#       print(name)
#       old_file_name = folder_name + '/' +name
#       new_file_name= folder_name + '/'+'DIV2K_'+name
#       os.rename(old_file_name, new_file_name)
print('end')
#img = cv2.imread('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR')