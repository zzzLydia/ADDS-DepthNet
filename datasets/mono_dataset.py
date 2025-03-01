from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 opt,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width

        self.interp = Image.ANTIALIAS

        if opt.only_depth_encoder:
            self.num_scales = 1
        else:

            self.num_scales = num_scales

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
# transform
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = True

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
    #introduce color_aug to input dict
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k or "color_n" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "color_n" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        
#change variable of filename(1 to 2), return 2 pic once


        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
#transform
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
#split each line in txt:  eg；day_train_all/0000000315

        line = self.filenames[index].split('/') #line=['day_train_all','0000000315']
        folder = line[0] # folder='day_train_all'

        # if len(line) == 3:
        frame_index = int(line[1]) #frame_index=0000000315
        # else:
        #     frame_index = 0

        istrain = folder.split('_')[1] # istrain=train

        if istrain == 'train':
            # add fake imgs (be paired), and be sure that folder is day, folder2 is night
            if folder[0] == 'd': # folder=day_train_all
                folder2 = folder + '_fake_night' # folder=day_train_all_fake_night
                flag = 0
            else:
                folder2 = folder + '_fake_day'
                tmp = folder
                folder = folder2
                folder2 = tmp
                flag = 1

            if len(line) == 3:
                side = line[2]
            else:
                side = None

            for i in self.frame_idxs: # default=[0, -1, 1]
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                    inputs[("color_n", i, -1)] = self.get_color(folder2, frame_index, other_side, do_flip)
                else:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip) # a color image
                    inputs[("color_n", i, -1)] = self.get_color(folder2, frame_index + i, side, do_flip) #frame index is 10-digit, not 0 -1 1
                    # plt.imsave('img.jpg',np.squeeze(np.array(inputs[("color", i, -1)])).astype(np.uint8))
                    # print(i)
# related to camera intrinsic, just use default
            # adjusting intrinsics to match each scale in the pyramid
            for scale in range(self.num_scales):
                K = self.K.copy()

                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                inv_K = np.linalg.pinv(K)
# introduction of K or inv_K
                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

            if do_color_aug:
                color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)
# introduction of color_aug
            self.preprocess(inputs, color_aug)
#delete image loaded from disk (because we have image size as opt)
            for i in self.frame_idxs:
                del inputs[("color", i, -1)]
                del inputs[("color_aug", i, -1)]
                del inputs[("color_n", i, -1)]
                del inputs[("color_n_aug", i, -1)]

            istrain = folder.split('_')[1] #folder='day_train_all' dayflag=0 nighrflag=1 here always0
            if istrain != 'train':
                if flag:
                    depth_gt = self.get_depth(folder2, frame_index, side, do_flip)
                else:
                    depth_gt = self.get_depth(folder, frame_index, side, do_flip)
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

            if "s" in self.frame_idxs:
                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side == "l" else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1

                inputs["stereo_T"] = torch.from_numpy(stereo_T)

            # plt.imsave('img.jpg', np.squeeze(np.array(inputs[("color", 0, 0)])).astype(np.uint8))
            # print(i)
        else: #day_val_451/0000030525 line=['day_val_451','0000030525']
            
            if folder[0] == 'd':    # folder=day_val_451
                folder2 = folder + '_fake_night'     # folder=day_val_451_fake_night
                flag = 0
            else:
                folder2 = folder + '_fake_day'
                tmp = folder
                folder = folder2
                folder2 = tmp
                flag = 1
                
                
            if len(line) == 3:
                side = line[2]
            else:
                side = None

            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                    inputs[("color_n", i, -1)] = self.get_color(folder2, frame_index, other_side, do_flip)
                    
                else:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                    inputs[("color_n", i, -1)] = self.get_color(folder2, frame_index + i, side, do_flip)

            # adjusting intrinsics to match each scale in the pyramid
# related to camera intrinsic, just use default

            for scale in range(self.num_scales):
                K = self.K.copy()

                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                inv_K = np.linalg.pinv(K)
# introduction of K or inv_K
                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

            if do_color_aug:
                color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)
# introduction of color_aug
            self.preprocess(inputs, color_aug)

            for i in self.frame_idxs:
                del inputs[("color", i, -1)]
                del inputs[("color_aug", i, -1)]
                del inputs[("color_n", i, -1)]
                del inputs[("color_n_aug", i, -1)]


            if flag:
                depth_gt = self.get_depth(folder2, frame_index, side, do_flip)
            else:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
                
                
                

            if "s" in self.frame_idxs:
                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side == "l" else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1

                inputs["stereo_T"] = torch.from_numpy(stereo_T)


        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
        
        
        
