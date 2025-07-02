from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
from torch.autograd import Variable
#from torch.autograd.gradcheck import zero_gradients
from loss_utils import *
from Attacker import CFG
from utility import (Average_meter, Count_converge_iter, Count_loss_iter,
                    _compare, accuracy, estimate_normal_via_ori_normal,
                    farthest_points_sample)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)



def main(cfg):

    print('=>Creating the directory of results')
    saved_dir = os.path.join('Results', cfg.source_model + '_npoint' + str(cfg.npoint) + '_' + str(cfg.cc_linf) + '_' + str(cfg.iter_max_steps) + '_' + str(cfg.CFG) + cfg.cls_loss_type + '_' + cfg.dis_loss_type + '_' + str(cfg.initial_const))
    print('==>Successfully created {}'.format(saved_dir))

    trg_dir = os.path.join(saved_dir, 'Mat')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)

    if cfg.id == 0:
        seed = 0
    else:
        seed = random.randint(1, 10000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data
    resample_num = -1

    from Provider.modelnet40 import ModelNet40
    test_dataset = ModelNet40(data_mat_file=cfg.data_dir_file, resample_num=resample_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    test_size = test_dataset.__len__()

    # model
    print('=>Loading model')
    model_path = os.path.join('Pretrained', cfg.source_model, str(cfg.npoint), 'model_best.pth.tar')
    if cfg.source_model == 'PointNet':
        from Model.PointNet import PointNet
        net = PointNet(cfg.classes, npoint=cfg.npoint).cuda()
    elif cfg.source_model == 'PointNetPP_ssg':
        from Model.PointNetPP_ssg import PointNet2ClassificationSSG
        net = PointNet2ClassificationSSG(use_xyz=True, use_normal=False).cuda()
    elif cfg.source_model == 'PointNetPP_msg':
        from Model.PointNetPP_msg import PointNet2ClassificationMSG
        net = PointNet2ClassificationMSG(use_xyz=True, use_normal=False).cuda()
    elif cfg.source_model == 'DGCNN':
        from Model.DGCNN import DGCNN_cls
        net = DGCNN_cls(k=20, emb_dims=cfg.npoint, dropout=0.5).cuda()
    elif cfg.source_model == 'PointConv':
        from Model.pointconv import PointConvClsSsg
        net = PointConvClsSsg().cuda()
    elif cfg.source_model == 'point_pn':
        from Model.point_pn import Point_PN_mn40
        net = Point_PN_mn40().cuda()
    elif cfg.source_model == 'pt1':
        from Model.pt1 import PointTransformerCls
        net = PointTransformerCls().cuda()
    else:
        assert False, 'Not support such source_model.'

    checkpoint = torch.load(model_path)
    #new_state_dict = {}
    #for k,v in checkpoint.items():
        #new_state_dict[k[7:]] = v
    #net.load_state_dict(new_state_dict)
    #net.load_state_dict(checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('==>Successfully load pretrained-model from {}'.format(model_path))

    test_acc = Average_meter()
    batch_vertice = []
    batch_faces_idx = []
    batch_gt_label = []

    num_attack_success = 0
    cnt_ins = test_dataset.start_index
    cnt_all = 0
    lc = 0
    lh = 0
    targeted = False
    num_attack_classes = 1

    for i, data in enumerate(test_loader):
        pc = data[0]
        normal = data[1]
        gt_labels = data[2]
        if pc.size(3) == 3:
            pc = pc.permute(0,1,3,2)
        if normal.size(3) == 3:
            normal = normal.permute(0,1,3,2)

        bs, l, _, n = pc.size()
        b = bs*l

        pc = pc.view(b, 3, n).cuda()
        normal = normal.view(b, 3, n).cuda()
        gt_target = gt_labels.view(-1).cuda()

        adv_pc, targeted_label, attack_success_indicator, best_attack_step, loss = CFG.attack(net, data, cfg, i, len(test_loader), saved_dir)
        eval_num = 1

        for _ in range(0,eval_num):
            with torch.no_grad():
                if adv_pc.size(2) > cfg.npoint:
                    eval_points = farthest_points_sample(adv_pc, cfg.npoint)
                else:
                    eval_points = adv_pc
                test_adv_output = net(eval_points)
            attack_success_iter = _compare(torch.max(test_adv_output,1)[1].data, targeted_label, gt_target.cuda(), targeted)

            try:
                attack_success += attack_success_iter
            except:
                attack_success = attack_success_iter
        saved_pc = adv_pc.cpu().clone().numpy()

        for k in range(b):
            if attack_success_indicator[k].item():
                num_attack_success += 1
                name = 'adv_' + str(cnt_ins+k//num_attack_classes) + '_gt' + str(gt_target[k].item()) + '_attack' + str(torch.max(test_adv_output,1)[1].data[k].item()) + '_expect' + str(targeted_label[k].item())

                sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'),
                {"adversary_point_clouds": saved_pc[k], 'gt_label': gt_target[k].item(), 'attack_label': torch.max(test_adv_output,1)[1].data[k].item()})

            cnt_ins = cnt_ins + bs
            cnt_all = cnt_all + b
        
        
        lc_loss = chamfer_loss(adv_pc, pc)
        lc += lc_loss
        lh_loss = hausdorff_loss(adv_pc, pc)
        lh += lh_loss

    print('attack success: {0:.2f}\n'.format(num_attack_success/float(cnt_all)*100))

    print('chamfer_loss: ', lc)
    print('hausdorff_loss: ', lh)

    print('saved_dir: {0}'.format(os.path.join(saved_dir)))

    print('Finish!')

    return saved_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack')
    #------------Model-----------------------
    parser.add_argument('--id', type=int, default=0, help='using fixed seed or not')
    parser.add_argument('--source_model', default='PointNet', type=str, metavar='source_model', help='source model')
    #------------Dataset-----------------------
    parser.add_argument('--data_dir_file', default='data/data.mat', type=str, help='path to datafile')
    #parser.add_argument('--dense_data_dir_file', default=None, type=str, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='B', help='batch_size (default: 1)')
    parser.add_argument('--npoint', default=1024, type=int, help='number of points ')
    #------------Attack-----------------------
    parser.add_argument('--initial_const', type=float, default=10, help='initial offset limitation ')
    #parser.add_argument('--binary_max_steps', type=int, default=1, help='')
    parser.add_argument('--iter_max_steps',  default=200, type=int, metavar='M', help='max steps')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--eval_num', type=int, default=1, help='')
    ## PF loss
    parser.add_argument('--PF_loss_weight', type=float, default=0, help='Loss For comparation')
    parser.add_argument('--pp', type=float, default=0.5)
    ## perturbation clip setting
    parser.add_argument('--cc_linf', type=float, default=0.18, help='Coefficient for infinity norm')
    ## eval metric
    parser.add_argument('--metric', default='Loss', type=str, help='[Loss | L2 ]')
    #------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    #-------input diversity-------
    parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
    parser.add_argument('--prob', default=1, type=float, help='probability')
    parser.add_argument('--mask_num', default=3, type=int, help='mask_num')
    parser.add_argument('--max_dropout_ratio', default=0.5, type=float, help='max_dropout_ratio')
    parser.add_argument('--input_a', default=0.5, type=float, help='input_a')
    parser.add_argument('--CFG', default=0.8, type=float, help='CFG')
    parser.add_argument('--cls_loss_type', default='CE', type=str, help='Margin | CE')
    parser.add_argument('--dis_loss_type', default='CD', type=str, help='CD | L2 | HD')
    parser.add_argument('--confidence', type=float, default=0, help='confidence for margin based attack method')

    cfg  = parser.parse_args()

    main(cfg)
