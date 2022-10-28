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
        if self.opt.predictive_mask:
            outputs['predictive_mask']=predictive_mask(feature)

        #pose
        output.update(predict_poses(input))
        
        #use pose, intrinsic -1/1 to a predict
        generate_images_pred(inputs, outputs)
        #output with day and night color predict

        losses_day=compute_losses(inputs, outputs, 'day')
        losses_night=compute_losses(inputs, outputs, 'night')
        
        
        loss=losses_day+losses_night
        
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.  
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))

        mean_errors_day = self.evaluate('day')
        mean_errors_night = self.evaluate('night')
        mean_errors_all = (mean_errors_day + mean_errors_night) /2
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_all.tolist()) + "\\\\")
        print("\n-> Done!")
    return mean_errors_day, mean_errors_night, mean_errors_all





    
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
    predictive_mask=networks.Depth.Decoder(pose_encoder.num_ch_enc, scales=range(0), num_output_channels=2)
    
    
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

    best_absrel = best_sqrel = best_rmse = best_rmse_log = np.inf
    best_a1 = best_a2 = best_a3 = 0
    best_epoch = 0
    for epoch in range(1, opt.epoch + 1):
        #print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')

        mean_errors_day, mean_errors_night, mean_errors_all = train(train_loader, test_loader,  model, optimizer, epoch, best_loss, device)

        mean_errors = []
        if best_rmse > mean_errors_all[2]:
            best_epoch = self.epoch
            best_absrel = mean_errors_all[0]
            best_sqrel = mean_errors_all[1]
            best_rmse = mean_errors_all[2]
            best_rmse_log = mean_errors_all[3]
            best_a1 = mean_errors_all[4]
            best_a2 = mean_errors_all[5]
            best_a3 = mean_errors_all[6]
        mean_errors.append(best_absrel)
        mean_errors.append(best_sqrel)
        mean_errors.append(best_rmse)
        mean_errors.append(best_rmse_log)
        mean_errors.append(best_a1)
        mean_errors.append(best_a2)
        mean_errors.append(best_a3)
        print('best results is %d epoch:' % best_epoch)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors) + "\\\\")
        print("\n-> Done!")
        
def predict_poses(inputs):
    """Predict poses between input frames for monocular sequences.
    """
    outputs = {}

    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in opt.frame_ids}

    for f_i in opt.frame_ids[1:]:
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
    for scale in opt.scales:
        disp = outputs[("disp", scale)]
        if opt.v1_multiscale:
            source_scale = scale
        else:
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        outputs[("depth", 0, scale)] = depth

        for i, frame_id in enumerate(self.opt.frame_ids[1:]):

  
            T = outputs[("cam_T_cam", 0, frame_id)]


            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](
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

            if not self.opt.disable_automasking:
                
                outputs[("color_n_identity", frame_id, scale)] = \
                    inputs[("color_n", frame_id, source_scale)]

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]
            
def compute_losses(inputs, outputs, is_night):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            if is_night:
                color = inputs[("color_n", 0, scale)]
                target = inputs[("color_n", 0, source_scale)]
            else:
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                if is_night:
                    pred = outputs[("color_n", frame_id, scale)]
                else:
                    pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if is_night:
                        pred = inputs[("color_n", frame_id, source_scale)]
                    else:
                        pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

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

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

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


def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if self.opt.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def evaluate(split='day'):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    model.eval()

    assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"


    pred_disps = []
    gt = []
    print("-> Computing predictions with size {}x{}".format(
        self.opt.width, self.opt.height))

    if split=='day':
        dataloader = self.val_day_loader
        val_split = 'val_day'
    elif split =='night':
        dataloader = self.val_night_loader
        val_split = 'val_night'

    with torch.no_grad():
        for data in dataloader:
            #input_color = data[("color", 0, 0)].cuda()

            if self.opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            
            output = model(data["color", 0, 0], data["color_n", 0, 0])

            pred_disp, _ = disp_to_depth(output, self.opt.min_depth, self.opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if self.opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            gt.append(np.squeeze(data['depth_gt'].cpu().numpy()))

    pred_disps = np.concatenate(pred_disps)
    gt = np.concatenate(gt)


    if self.opt.save_pred_disps:
        output_path = os.path.join(
            self.opt.load_weights_folder, "disps_{}_split.npy".format(self.opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if self.opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        self.opt.disable_median_scaling = True
        self.opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if self.opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # # range 60m
        # mask2 = gt_depth<=40
        # pred_depth = pred_depth[mask2]
        # gt_depth = gt_depth[mask2]

        pred_depth *= self.opt.pred_depth_scale_factor
        if not self.opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(self.compute_errors(gt_depth, pred_depth))

    if not self.opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    with open(self.result_path, 'a') as f: 
        for i in range(len(mean_errors)):
            f.write(str(mean_errors[i])) #
            f.write('\t')
        f.write("\n")        

    f.close()  

#         self.log_val(val_split, data, output)

    self.set_train()
    return mean_errors

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
