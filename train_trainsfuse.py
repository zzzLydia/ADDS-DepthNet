from __future__ import absolute_import, division, print_function
import os
import argparse
import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import json
from lib.TransFuse import TransFuse_S
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
from IPython import embed
import random

from lib import TransFuse

STEREO_SCALE_FACTOR = 5.4


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #self.parser.add_argument('--epoch', type=int, default=25, help='epoch number')
        self.parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
        self.parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
        self.parser.add_argument('--train_path', type=str,
                            default='data/', help='path to train dataset')
        self.parser.add_argument('--test_path', type=str,
                            default='data/', help='path to test dataset')
        self.parser.add_argument('--train_save', type=str, default='TransFuse_S')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
        self.parser.add_argument('--device', type=str, default="cuda:1", help='gpu name')
        self.parser.add_argument("--data_path",type=str,help="path to the training data", default="../../../oxford_processing_new")
        self.parser.add_argument('--scheduler_step_size', type=float, default=0.1)
        self.parser.add_argument('--num_epochs', type=int, default=50, help='beta2 of adam optimizer')
        self.parser.add_argument('--scales', type=int, default=[0], help='beta2 of adam optimizer')
        self.parser.add_argument('--v1_multiscale', action="store_true", help='beta2 of adam optimizer')
        self.parser.add_argument('--min_depth', type=float, default=0.1, help='beta2 of adam optimizer')
        self.parser.add_argument('--max_depth', type=float, default=100.0, help='beta2 of adam optimizer')
        self.parser.add_argument('--disable_automasking', action="store_true", help='beta2 of adam optimizer')
        self.parser.add_argument("--no_ssim",action="store_true",help="if set, disables ssim in the loss")
        self.parser.add_argument("--avg_reprojection",action="store_true", help="if set, uses average reprojection loss")
        self.parser.add_argument("--disparity_smoothness",type=float,default=1e-3, help="disparity smoothness weight")
        self.parser.add_argument("--post_process",action="store_true",help="if set will perform the flipping post processing "
                                      "from the original monodepth paper")
        self.parser.add_argument("--disable_median_scaling",action="store_true",help="if set disables median scaling in evaluation")
        self.parser.add_argument("--pred_depth_scale_factor",type=float,default=1, help="if set multiplies predictions by this number")
        
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
    
options = Options()
opts = options.parse()



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
class Trainer:
    def __init__(self, options):
        self.opt=options
        self.data_path="train_image"
        self.start_time=time.time()
        self.height=256
        self.width=512
        self.frame_ids=[0, -1, 1]
        self.parameters_to_train=[]
        self.log_path=os.path.join(os.path.join('/home/omega/Documents/Zezheng/ADDS_transfuse/ADDS-DepthNet-main', "tmp"),'transfuse')
        self.result_path=os.path.join(self.log_path, 'result.txt')

        self.model = TransFuse_S(pretrained=True).to(self.opt.device)
        self.parameters_to_train+=list(self.model.parameters())
        self.pose_encoder=networks.ResnetEncoder_pose(num_layers=18, pretrained=True, num_input_images=2).to(self.opt.device)
        self.parameters_to_train+=list(self.pose_encoder.parameters())
        self.pose_decoder=networks.PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(self.opt.device)
        self.parameters_to_train+=list(self.pose_decoder.parameters())
#         self.predictive_mask=networks.DepthDecoder(pose_encoder.num_ch_enc, scales=range(0), num_output_channels=2)
#         self.parameters_to_train+=list(self.model.parameters())
        self.backproject_depth=BackprojectDepth(self.opt.batch_size, 256, 512).to(self.opt.device)
        self.project_3d=Project3D(self.opt.batch_size, 256, 512).to(self.opt.device)
        self.ssim=SSIM().to(self.opt.device)
        # ---- build models ----
        self.num_scales=1
        
        self.optimizer = torch.optim.Adam(self.parameters_to_train, self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        #print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.opt.scheduler_step_size, 0.1)


        fpath = os.path.join(os.path.dirname(__file__), "splits", 'oxford_day', "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_day_filenames = readlines(fpath.format("val_day"))
        val_night_filenames = readlines(fpath.format("val_night"))
        
        train_dataset = datasets.KITTIRAWDataset(self.opt,
            self.data_path, train_filenames, self.height,self. width,
            self.frame_ids, num_scales=1, is_train=True)
        self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, shuffle=True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)


        val_day_dataset = datasets.KITTIRAWDataset(self.opt, self.data_path, val_day_filenames, self.height, self.width,
            [0], num_scales=1, is_train=False)
        self.val_day_loader = DataLoader(
            val_day_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter_day = iter(self.val_day_loader)


        val_night_dataset = datasets.KITTIRAWDataset(self.opt, self.data_path, val_night_filenames, self.height, self.width,
            [0], num_scales=1, is_train=False)
        self.val_night_loader = DataLoader(
            val_night_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter_night = iter(self.val_night_loader)

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
    def train(self):
        
#             print("#"*20, "Start Training", "#"*20)
#         total_step = len(train_loader)
        self.step=0

        best_absrel = best_sqrel = best_rmse = best_rmse_log = np.inf
        best_a1 = best_a2 = best_a3 = 0
        best_epoch = 0
        for self.epoch in range(self.opt.num_epochs):
            
            mean_errors_day, mean_errors_night, mean_errors_all = self.run_epoch()

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
    def run_epoch(self):
        self.scheduler.step()
        print('Training')
        self.model.train()
        self.pose_encoder.train()
        self.pose_decoder.train()

        for i, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.opt.device)
                
            before_op_time=time.time()
#             print(inputs["color_aug", 0, 0].shape)
#             print(inputs["color_n_aug", 0, 0].shape)
            outputs = {}
            #d3, d2, output[("disp", i)] = model(inputs["color_aug", 0, 0], inputs["color_n_aug", 0, 0])
            d3, d2, outputs[('disp', 0)] = self.model(inputs["color_aug", 0, 0], inputs["color_n_aug", 0, 0])
#             print(outputs[('disp', 0)].shape)

            #predictive mask
    #         if self.opt.predictive_mask:
    #             outputs['predictive_mask']=predictive_mask(feature)

            #pose
            outputs.update(self.predict_poses(inputs))

            #use pose, intrinsic -1/1 to a predict
            self.generate_images_pred(inputs, outputs)
            #output with day and night color predict

            losses_day=self.compute_losses(inputs, outputs, 'day')
            losses_night=self.compute_losses(inputs, outputs, 'night')
#             print(losses_day)
#             print(losses_night)

            loss=losses_day['loss']+losses_night['loss']

            loss.backward() 
#             torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            duration=time.time()-before_op_time

            if i % 10 == 0:
                self.log_time(i, duration, losses_day["loss"].cpu().data+losses_night["loss"].cpu().data)
        self.step+=1
        mean_errors_day = self.evaluate('day')
        mean_errors_night = self.evaluate('night')
        mean_errors_all = (mean_errors_day + mean_errors_night) /2
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_all.tolist()) + "\\\\")
        print("\n-> Done!")
        return mean_errors_day, mean_errors_night, mean_errors_all



    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
  
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

        for f_i in self.frame_ids[1:]:
                        # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]

            axisangle, translation = self.pose_decoder(pose_inputs)
#             print(axisangle.shape)
#             print(translation.shape)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation
            
            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs



    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.height, self.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):


                T = outputs[("cam_T_cam", 0, frame_id)]


                cam_points = self.backproject_depth(
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d(
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

                    outputs[("color_n_identity", frame_id, scale)] =inputs[("color_n", frame_id, source_scale)]

                    outputs[("color_identity", frame_id, scale)] =inputs[("color", frame_id, source_scale)]
                    
                    
                    
            
    def compute_losses(self, inputs, outputs, is_night):
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

                for frame_id in self.frame_ids[1:]:
                    if is_night:
                        pred = outputs[("color_n", frame_id, scale)]
                    else:
                        pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.frame_ids[1:]:
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
                        identity_reprojection_loss.shape).to(self.opt.device) * 0.00001

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


    def compute_reprojection_loss(self, pred, target):
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

    def evaluate(self, split='day'):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = self.opt.min_depth
        MAX_DEPTH = self.opt.max_depth
        self.model.eval()
        self.pose_encoder.eval()
        self.pose_decoder.eval()

#         assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1,         "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"


        pred_disps = []
        gt = []
        print("-> Computing predictions with size {}x{}".format(
            self.width, self.height))

        if split=='day':
            dataloader = self.val_day_loader
            val_split = 'val_day'
        elif split =='night':
            dataloader = self.val_night_loader
            val_split = 'val_night'

        with torch.no_grad():
            for data in dataloader:
                #input_color = data[("color", 0, 0)].cuda()
                for key, ipt in data.items():
                    data[key] = ipt.to(self.opt.device)

                if self.opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                d1,d2,outputs = self.model(data["color", 0, 0], data["color_n", 0, 0])
#                 print(outputs.shape)
                pred_disp, _ = disp_to_depth(outputs, self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if self.opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                gt.append(np.squeeze(data['depth_gt'].cpu().numpy()))

        pred_disps = np.concatenate(pred_disps)
        gt = np.concatenate(gt)


#         if self.opt.save_pred_disps:
#             output_path = os.path.join(
#                 self.opt.load_weights_folder, "disps_{}_split.npy".format(self.opt.eval_split))
#             print("-> Saving predicted disparities to ", output_path)
#             np.save(output_path, pred_disps)

        # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        print("-> Evaluating")

#         if self.opt.eval_stereo:
#             print("   Stereo evaluation - "
#                   "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
#             self.opt.disable_median_scaling = True
#             self.opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
#         else:
        print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):

            gt_depth = gt[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

#             if self.opt.eval_split == "eigen":
#                 mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

#                 crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
#                                  0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
#                 crop_mask = np.zeros(mask.shape)
#                 crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
#                 mask = np.logical_and(mask, crop_mask)

#             else:
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

#         with open(self.result_path, 'a') as f: 
#             for i in range(len(mean_errors)):
#                 f.write(str(mean_errors[i])) #
#                 f.write('\t')
#             f.write("\n")        

#         f.close()  

    #         self.log_val(val_split, data, output)

        self.model.train()
        self.pose_encoder.train()
        self.pose_decoder.train()
        return mean_errors

    def compute_errors(self, gt, pred):
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
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss))

    
if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
