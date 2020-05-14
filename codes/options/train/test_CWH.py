# import cv2 as cv
# import numpy as np
# import scipy.ndimage
# # import scipy.misc
# # import scipy
# # #import scipy
# from imageio import imwrite
# from matplotlib.image import imread
# #from scipy.misc import imresize
# from PIL import Image
# import numpy as np
# from PIL import Image

from codes.data.util import modcrop

# path = "/media/lcc/CWH/LR/gray/HR/ILSVRC2012_val_00000760.JPEG"
# scale = 4
# image = imread(path)
#
# # if image.shape[2] == 4:
# #   im = Image.open(path).convert("RGB")
# #   #print(im.shape)
# # label_ = modcrop(im, scale)
# # print(label_.shape)
# # im = np.array(Image.fromarray(label_).resize((int(612/4), int(460 / 4))))
# #im_LR = imresize(label_, 1/4, 'bicubic')
# # Must be normalized
# # image = image / 255.
# # label_ = label_ / 255.
# # #
# # input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
# imwrite('/media/lcc/CWH/LR/HR/1.JPEG', im)
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self,img_path,selected_layer):
        self.img_path=img_path
        self.selected_layer=selected_layer
        self.pretrained_model = models.vgg16(pretrained=True).features

    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        print(input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features=self.get_feature()
        print(features.shape)

        feature=features[:,0,:,:]
        print(feature.shape)

        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)

        return feature

    def save_feature_to_img(self):
        #to numpy
        feature=self.get_single_feature()
        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        print(feature[0])

        cv2.imwrite('/home/lcc/CWH/img.jpg',feature)




if __name__=='__main__':
    # get class
    myClass=FeatureVisualization('/home/lcc/CWH/Dataset2/Flickr2K/2K_train_HR_new/error/HR/002060.png',5)
    print (myClass.pretrained_model)

    myClass.save_feature_to_img()

print("end")

# input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
