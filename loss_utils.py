from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time
import copy
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#from torch.autograd.gradcheck import zero_gradients
import scipy.io as io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Model'))
from utility import _normalize

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)



def chamfer_loss(adv_pc, ori_pc):
    adv_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1)
    ori_KNN = knn_points(ori_pc.permute(0,2,1), adv_pc.permute(0,2,1), K=1)
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1) + ori_KNN.dists.contiguous().squeeze(-1).mean(-1)
    return dis_loss

def hausdorff_loss(adv_pc, ori_pc):
    #dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    #hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0] #[b]
    return hd_loss



def sample(delta, pp):
    b, s, n = delta.size()
    only_add_one_mask = torch.from_numpy(np.random.choice([0, 1], size=(b,s,n), p=[1 - pp, pp])).cuda()


    leave_one_mask = 1 - only_add_one_mask

    only_add_one_perturbation = delta * only_add_one_mask
    leave_one_out_perturbation = delta * leave_one_mask

    return only_add_one_perturbation, leave_one_out_perturbation

def get_features(
    model,
    x,
    perturbation,
    leave_one_out_perturbation,
    only_add_one_perturbation,
):

    outputs = model(x + perturbation)
    leave_one_outputs = model(x + leave_one_out_perturbation)
    only_add_one_outputs = model(x + only_add_one_perturbation)

    return (outputs, leave_one_outputs, only_add_one_outputs)

