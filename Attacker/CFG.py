from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time
import copy
import ipdb
import numpy as np
import open3d as o3d
from pytorch3d.ops import knn_points, knn_gather
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#from torch.autograd.gradcheck import zero_gradients


from utility import _compare, farthest_points_sample
from loss_utils import *


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def offset_proj(offset, ori_pc, ori_normal, project='dir'):
    # offset: shape [b, 3, n], perturbation offset of each point
    # normal: shape [b, 3, n], normal vector of the object

    condition_inner = torch.zeros(offset.shape).cuda().byte()

    intra_KNN = knn_points(offset.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    normal_len = (normal**2).sum(1, keepdim=True).sqrt()
    normal_len_expand = normal_len.expand_as(offset) #[b, 3, n]

    # add 1e-6 to avoid dividing by zero
    offset_projected = (offset * normal / (normal_len_expand + 1e-6)).sum(1,keepdim=True) * normal / (normal_len_expand + 1e-6)

    # let perturb be the projected ones
    offset = torch.where(condition_inner, offset, offset_projected)

    return offset

def find_offset(ori_pc, adv_pc):
    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    knn_pc = knn_gather(ori_pc.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    real_offset =  adv_pc - knn_pc

    return real_offset

def norm_l2_loss(adv_pc, ori_pc):
    return ((adv_pc - ori_pc)**2).sum(1).sum(1)
def lp_clip(offset, cc_linf):
    lengths = (offset**2).sum(1, keepdim=True).sqrt() #[b, 1, n]
    lengths_expand = lengths.expand_as(offset) # [b, 3, n]

    condition = lengths > 1e-6
    offset_scaled = torch.where(condition, offset / lengths_expand * cc_linf, torch.zeros_like(offset))

    condition = lengths < cc_linf
    offset = torch.where(condition, offset, offset_scaled)

    return offset

def _forward_step(net, pc_ori, input_curr_iter, normal_ori, target, scale_const, cfg, targeted, delta):
    b,_,n=input_curr_iter.size()
    output_curr_iter = net(input_curr_iter)

    if cfg.cls_loss_type == 'Margin':
        target_onehot = torch.zeros(target.size() + (cfg.classes,)).cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        fake = (target_onehot * output_curr_iter).sum(1)
        other = ((1. - target_onehot) * output_curr_iter - target_onehot * 10000.).max(1)[0]
        cls_loss = torch.clamp(fake - other + cfg.confidence, min=0.)  # equiv to max(..., 0.)
    else:
        cls_loss = - nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
    info = 'cls_loss: {0:6.4f}\t'.format(cls_loss.mean().item())

    if cfg.dis_loss_type == 'CD':
        dis_loss = chamfer_loss(input_curr_iter, pc_ori)
    
        constrain_loss = dis_loss
        info = info + 'cd_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'L2':
        dis_loss = norm_l2_loss(input_curr_iter, pc_ori)
        constrain_loss = dis_loss
        info = info + 'l2_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'HD':
        #print('HD')
        dis_loss = hausdorff_loss(input_curr_iter, pc_ori)
        constrain_loss = dis_loss
        info = info + 'hd_loss : {0:6.4f}\t'.format(dis_loss.mean().item())
    else:
        dis_loss = chamfer_loss(input_curr_iter, pc_ori)
        constrain_loss = dis_loss  * 0.0000001


    #input diversity loss
    
    input_diversity_loss_num=0
    r = np.random.rand(1)
    if cfg.beta > 0 and r < cfg.prob:

        for i in range(int(cfg.mask_num)):
            inputdiversity = random_point_dropout(pc_ori, max_dropout_ratio=cfg.max_dropout_ratio)
            inputdiversity_out = net(inputdiversity + delta)
            inputdiversity_out_c = copy.deepcopy(inputdiversity_out.detach())
            inputdiversity_out_c[:, target] = -np.inf
            other_max1 = inputdiversity_out_c.max(1)[1].item()

            input_diversity_loss = ((inputdiversity_out[:, other_max1] - inputdiversity_out[:, target.item()])**2)
            input_diversity_loss_num = input_diversity_loss_num+ cfg.beta * input_diversity_loss
        constrain_loss = constrain_loss +  input_diversity_loss_num / cfg.mask_num
        
        #input loss
        #pc_ori_attack = net(pc_ori + delta)
        #pc_ori_attack_c = copy.deepcopy(pc_ori_attack.detach())
        #pc_ori_attack_c[:, target] = -np.inf
        #pc_ori_other_max1 = pc_ori_attack_c.max(1)[1].item()

        #input_loss = ((pc_ori_attack[:, pc_ori_other_max1] - pc_ori_attack[:, target.item()])**2)
        #constrain_loss = constrain_loss + cfg.input_a * input_loss
    
    #CFG_loss
    if cfg.CFG != 0:
        
        net.zero_grad()
        img_temp_i = pc_ori.clone()
        out,y = net.features_grad(img_temp_i)
        grad_temp = torch.autograd.grad(out, y, grad_outputs=torch.ones_like(out))[0]
        x_cle = pc_ori.detach()
        x_adv = pc_ori.clone()
        mid_feature = net.layer2_features(x_adv+ delta)
        CFGloss = torch.sum(torch.abs(grad_temp * mid_feature)) * 0.01
        info = info+'CFG_loss : {0:6.4f}\t'.format(CFGloss.item())
        constrain_loss = constrain_loss +  cfg.CFG * CFGloss
    # PF_loss
    if cfg.PF_loss_weight != 0:
        only_add_one_perturbation, leave_one_out_perturbation = sample(delta, cfg.pp)

        (outputs, leave_one_outputs, only_add_one_outputs) = get_features(net, pc_ori, delta,
                                      leave_one_out_perturbation,
                                      only_add_one_perturbation)

        outputs_c = copy.deepcopy(outputs.detach())
        outputs_c[:, target] = -np.inf
        other_max = outputs_c.max(1)[1].item()
        pf_loss = PFLoss(target=other_max, label=target)

        average_pairwise = pf_loss(
            outputs, leave_one_outputs, only_add_one_outputs)

        constrain_loss = constrain_loss + cfg.PF_loss_weight * average_pairwise
        info = info+'pf_loss : {0:6.4f}\t'.format(average_pairwise.item())
    else:
        pf_loss = 0 
        
    scale_const = scale_const.float().cuda()
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return output_curr_iter, loss, loss_n, cls_loss, dis_loss, constrain_loss, info


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''

    batch_pc = batch_pc.permute(0,2,1)
    a = batch_pc.clone()
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random()*max_dropout_ratio  # set the ratio for dropout with the max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]  #select the index of points to be dropped 
        if len(drop_idx) > 0:
            
            a[b,drop_idx,:] = batch_pc[b,0,:]   #simulate drop operation  
        
    a = a.permute(0,2,1)
    return a




def attack(net, input_data, cfg, i, loader_len, saved_dir=None):

    step_print_freq = 50
    targeted = False

    pc = input_data[0]
    normal = input_data[1]
    gt_labels = input_data[2]
    if pc.size(3) == 3:
        pc = pc.permute(0,1,3,2)
    if normal.size(3) == 3:
        normal = normal.permute(0,1,3,2)

    bs, l, _, n = pc.size()
    b = bs*l

    pc_ori = pc.view(b, 3, n).cuda()
    normal_ori = normal.view(b, 3, n).cuda()
    gt_target = gt_labels.view(-1)
    target = gt_target.cuda()


    lower_bound = torch.ones(b) * 0
    scale_const = torch.ones(b) * cfg.initial_const
    upper_bound = torch.ones(b) * 1e10

    best_loss = [1e10] * b
    best_attack = torch.ones(b, 3, n).cuda()
    best_attack_step = [-1] * b
    best_attack_BS_idx = [-1] * b
    all_loss_list = [[-1] * b] * cfg.iter_max_steps
    for search_step in range(cfg.binary_max_steps):
        iter_best_loss = [1e10] * b
        iter_best_score = [-1] * b
        constrain_loss = torch.ones(b) * 1e10
        attack_success = torch.zeros(b).cuda()

        input_all = None

        for step in range(cfg.iter_max_steps):
            if step == 0:
                offset = torch.zeros(b, 3, n).cuda()
                nn.init.normal_(offset, mean=0, std=1e-3)
                offset.requires_grad_()

                optimizer = optim.Adam([offset], lr=cfg.lr)
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)
                periodical_pc = pc_ori.clone()

            ##input diversity
            #r = np.random.rand(1)
            #if cfg.beta > 0 and r < cfg.prob:
                #periodical_pc = random_point_dropout(periodical_pc, max_dropout_ratio=0.875)
            
            
            input_all = periodical_pc + offset

            if (input_all.size(2) > cfg.npoint) and (not cfg.is_partial_var) and cfg.is_subsample_opt:
                input_curr_iter = farthest_points_sample(input_all, cfg.npoint)
            else:
                input_curr_iter = input_all

            with torch.no_grad():
                for k in range(b):
                    if input_curr_iter.size(2) < input_all.size(2):
                        batch_k_pc = farthest_points_sample(torch.cat([input_all[k].unsqueeze(0)]*cfg.eval_num), cfg.npoint)
                        batch_k_adv_output = net(batch_k_pc)
                        attack_success[k] = _compare(torch.max(batch_k_adv_output,1)[1].data, target[k], gt_target[k], targeted).sum() > 0.5 * cfg.eval_num
                        output_label = torch.max(batch_k_adv_output,1)[1].mode().values.item()
                    else:
                        adv_output = net(input_curr_iter[k].unsqueeze(0))
                        output_label = torch.argmax(adv_output).item()
                        attack_success[k] = _compare(output_label, target[k], gt_target[k].cuda(), targeted).item()

                    metric = constrain_loss[k].item()

                    if attack_success[k] and (metric <best_loss[k]):
                        best_loss[k] = metric
                        best_attack[k] = input_all.data[k].clone()
                        best_attack_BS_idx[k] = search_step
                        best_attack_step[k] = step
                    if attack_success[k] and (metric <iter_best_loss[k]):
                        iter_best_loss[k] = metric
                        iter_best_score[k] = output_label

            _, loss, loss_n, cls_loss, dis_loss, constrain_loss, info = _forward_step(net, pc_ori, input_curr_iter, normal_ori, target, scale_const, cfg, targeted, offset)

            all_loss_list[step] = loss_n.detach().tolist()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                proj_offset = offset_proj(offset, pc_ori, normal_ori)
                offset.data = proj_offset.data

            if cfg.cc_linf != 0:
                with torch.no_grad():
                    proj_offset = lp_clip(offset, cfg.cc_linf)
                    offset.data = proj_offset.data

            info = '[{5}/{6}][{0}/{1}][{2}/{3}] \t loss: {4:6.4f}\t'.format(search_step+1, cfg.binary_max_steps, step+1, cfg.iter_max_steps, loss.item(), i, loader_len) + info

            if step % step_print_freq == 0 or step == cfg.iter_max_steps - 1:
                print(info)

        # adjust the scale constants
        for k in range(b):
            if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and iter_best_score[k] != -1:
                lower_bound[k] = max(lower_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                else:
                    scale_const[k] *= 2
            else:
                upper_bound[k] = min(upper_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

    return best_attack, target, (np.array(best_loss)<1e10), best_attack_step, all_loss_list  #best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b], all_loss_list:[iter_max_steps, b]