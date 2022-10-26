from __future__ import absolute_import, division, print_function
import os
import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
import random

from lib import TransFuse

STEREO_SCALE_FACTOR = 5.4


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse
        
def train(train_loader, test_loader, model, optimizer, epoch, best_loss, device):
    
    model.train()

    for i, inputs in enumerate(train_loader, start=1):
        inputs=inputs.to(device)
        d3, d2, output[("disp", i)] = model(inputs["color_aug", 0, 0], inputs["color_n_aug", 0, 0])
        #predictive mask
        #pose
        output.update(predict_poses(input))
        
        #use pose, intrinsic -1/1 to a predict
        generate_images_pred(inputs, outputs)
        

        loss3 = 
      
        









































































    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=25, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--device', type=str, default="cuda:1", help='gpu name')


    opt = parser.parse_args()

    # ---- build models ----
    data_path="../../../oxford_processing_new"
    height=256
    width=512
    frame_ids=[0, -1, 1]
    
    model = TransFuse_S(pretrained=True).to(opt.device)
    
    pose_encoder=networks.ResnetEncoder_Pose(num_layers=18, pretrained=True, num_input_images=2).to(opt.device)
    pose_decoder=networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(opt.device)
    
    backproject_depth=BackprojectDepth(opt.batch_size, 256, 512).to(opt.device)
    project_3d=Project3D(opt.batch_size, 256, 512).to(opt.device)
    
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
    #print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
    
    
    
    fpath = os.path.join(os.path.dirname(__file__), "splits", oxford_day, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    val_day_filenames = readlines(fpath.format("val_day"))
    val_night_filenames = readlines(fpath.format("val_night"))
    
    train_dataset = datasets.KITTIRAWDataset(options,
            data_path, train_filenames, height, width,
            frame_ids, num_scales=0, is_train=True)
    train_loader = DataLoader(
            train_dataset, opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    

    val_day_dataset = datasets.KITTIRAWDataset(opt.data_path, val_day_filenames, height, width,
        [0], num_scales=0, is_train=False)
    val_day_loader = DataLoader(
        val_day_dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_iter_day = iter(val_day_loader)


    val_night_dataset = datasets.KITTIRAWDataset(opt.data_path, val_night_filenames, height, width,
        [0], num_scales=0, is_train=False)
    val_night_loader = DataLoader(
        val_night_dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_iter_night = iter(val_night_loader)
    
    
    
    

    print("#"*20, "Start Training", "#"*20)
    total_step = len(train_loader)
    device=opt.device

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        #print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        best_loss = train(train_loader, test_loader,  model, optimizer, epoch, best_loss, device)
        
def predict_poses(inputs):
    """Predict poses between input frames for monocular sequences.
    """
    outputs = {}

    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

    for f_i in self.opt.frame_ids[1:]:
                    # To maintain ordering we always pass frames in temporal order
        if f_i < 0:
            pose_inputs = [pose_feats[f_i], pose_feats[0]]
        else:
            pose_inputs = [pose_feats[0], pose_feats[f_i]]

        pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]

        axisangle, translation = pose_decoder(pose_inputs)
        outputs[("axisangle", 0, f_i)] = axisangle
        outputs[("translation", 0, f_i)] = translation

        # Invert the matrix if the frame id is negative
        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    return outputs



def generate_images_pred(inputs, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    for scale in self.opt.scales:
        disp = outputs[("disp", scale)]

        source_scale = scale


        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        outputs[("depth", 0, scale)] = depth

        for i, frame_id in enumerate(self.opt.frame_ids[1:]):


            T = outputs[("cam_T_cam", 0, frame_id)]

            cam_points = backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = project_3d[source_scale](
                cam_points, inputs[("K", source_scale)], T)

            outputs[("sample", frame_id, scale)] = pix_coords


            outputs[("color_n", frame_id, scale)] = F.grid_sample(
                inputs[("color_n", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")

            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")
            
            
def compute_losses(self, inputs, outputs, is_night):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    losses = {}
    total_loss = 0

    for scale in self.opt.scales:
        loss = 0
        reprojection_losses = []

        source_scale = scale


        disp = outputs[("disp", scale)]
        if is_night:
            color = inputs[("color_n", 0, scale)]
            target = inputs[("color_n", 0, source_scale)]
        else:
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

        for frame_id in self.opt.frame_ids[1:]:
            pred = outputs[("color", frame_id, scale)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not self.opt.disable_automasking:
        elif self.opt.predictive_mask:
            # use the predicted mask
            mask = outputs["predictive_mask"]["disp", scale]
            if not self.opt.v1_multiscale:
                mask = F.interpolate(
                    mask, [self.opt.height, self.opt.width],
                    mode="bilinear", align_corners=False)

            reprojection_losses *= mask

            # add a loss pushing mask to 1 (using nn.BCELoss for stability)
            weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            loss += weighting_loss.mean()


        reprojection_loss = reprojection_losses.mean(1, keepdim=True)

        combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)


        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= self.num_scales
    losses["loss"] = total_loss
    return losses
