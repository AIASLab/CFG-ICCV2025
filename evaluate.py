from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utility import farthest_points_sample
from utility import (Average_meter, Count_converge_iter, Count_loss_iter,
                    _compare, accuracy, estimate_normal_via_ori_normal,
                    farthest_points_sample)


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
            
            
def main():

    # data 
    test_dataset = ModelNet40(cfg.datadir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                                              num_workers=cfg.num_workers, pin_memory=True)
    test_size = test_dataset.__len__()

    # model
    model_path = os.path.join('Pretrained', cfg.target_model, str(cfg.npoint), 'model_best.pth.tar')
    if cfg.target_model == 'PointNet':
        from Model.PointNet import PointNet
        net = PointNet(cfg.classes, npoint=cfg.npoint).cuda()
    elif cfg.target_model == 'PointNetPP_ssg':
        from Model.PointNetPP_ssg import PointNet2ClassificationSSG
        net = PointNet2ClassificationSSG(use_xyz=True, use_normal=False).cuda()
    elif cfg.target_model == 'PointNetPP_msg':
        from Model.PointNetPP_msg import PointNet2ClassificationMSG
        net = PointNet2ClassificationMSG(use_xyz=True, use_normal=False).cuda()
    elif cfg.target_model == 'DGCNN':
        from Model.DGCNN import DGCNN_cls
        net = DGCNN_cls(k=20, emb_dims=cfg.npoint, dropout=0.5).cuda()
    elif cfg.target_model == 'pointconv':
        from Model.pointconv import PointConvDensityClsSsg
        net = PointConvDensityClsSsg().cuda()
    elif cfg.target_model == 'PCT':
        from Model.PCT import Pct
        net = Pct().cuda()
    elif cfg.target_model == 'pointcnn':
        from Model.pointcnn import pointcnn
        net = pointcnn().cuda()
    elif cfg.target_model == 'curvenet':
        from Model.curvenet import curvenet
        net = curvenet().cuda()
    elif cfg.target_model == 'point_pn':
        from Model.point_pn import Point_PN_mn40
        net = Point_PN_mn40().cuda()
    elif cfg.target_model == 'pt1':
        from Model.pt1 import PointTransformerCls
        net = PointTransformerCls().cuda()
    elif cfg.target_model == 'ccn':
        from Model.ccn import ccn
        net = ccn(k=20, emb_dims=cfg.npoint, dropout=0.5).cuda()
    elif cfg.target_model == 'PointNet-AT':
        from Model.PointNet import PointNet
        net = PointNet(cfg.classes, npoint=cfg.npoint).cuda()
    else:
        assert False, 'Not support such target_model.'

    checkpoint = torch.load(model_path)
    #new_state_dict = {}
    #for k,v in checkpoint.items():
        #new_state_dict[k[7:]] = v
    #net.load_state_dict(new_state_dict)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('\nSuccessfully load pretrained-model from {}\n'.format(model_path))

    cnt = 0
    num_pred_success = 0

    for i, (adv_pc, gt_label, attack_label) in enumerate(test_loader):
        b = adv_pc.size(0)
        assert b == 1
        cnt += 1

        if adv_pc.size(2) > cfg.npoint:
            # adv_pc = adv_pc[:,:,:cfg.npoint]
            adv_pc = farthest_points_sample(adv_pc.cuda(), cfg.npoint)

        with torch.no_grad():
            attack_output = net(adv_pc.cuda())

        pred_success = (torch.max(attack_output, 1)[1].data.cpu() == gt_label.view(-1)).sum() # [b, num_class]
        #pred_success = (torch.max(attack_output, 1)[1].data.cpu() != gt_label.view(-1)) # [b, num_class]
        #pred_success = _compare(torch.max(attack_output,1)[1].data, targeted_label, gt_label.view(-1)).cuda(), targeted)
        num_pred_success += pred_success

    

    final_acc = num_pred_success.item() / float(test_loader.dataset.__len__()) * 100
    print('\nfinal attack success: {0:.2f}'.format(100 - final_acc))

    print('\n Finished!')


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'Model'))
    sys.path.append(os.path.join(ROOT_DIR, 'Lib'))
    sys.path.append(os.path.join(ROOT_DIR, 'Provider'))
    from loss_utils import *
    from Provider.eva_modelnet40 import ModelNet40

    parser = argparse.ArgumentParser(description='Point Cloud Evaluate')
    # ------------Dataset-----------------------
    parser.add_argument('--datadir', default='./result/DGCNN18', type=str, metavar='DIR',
                        help='path to adv. dataset')
    parser.add_argument('--npoint', default=1024, type=int, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    # ------------Model-----------------------
    parser.add_argument('--target_model', default='PointNetPP_msg', type=str, metavar='target_model', help='target_model')
    # ------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--print_freq', default=50, type=int, help='')

    cfg = parser.parse_args()
    print(cfg)

    assert cfg.datadir[-1] != '/'

    main()
