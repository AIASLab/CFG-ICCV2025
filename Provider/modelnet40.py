import os
import sys
import numpy as np
from random import choice
from scipy.io import loadmat

import torch
from torch.utils.data.dataloader import default_collate


class ModelNet40():
    def __init__(self, data_mat_file='data.mat', attack_label='Untarget', resample_num=-1, is_half_forward=False):
        self.data_root = data_mat_file
        self.attack_label = attack_label
        self.is_half_forward = is_half_forward

        if not os.path.isfile(self.data_root):
            assert False, 'No exists .mat file!'

        dataset = loadmat(self.data_root)
        data = torch.FloatTensor(dataset['data'])
        normal = torch.FloatTensor(dataset['normal'])
        label = dataset['label']

        if resample_num>0:
            tmp_data_set = []
            tmp_normal_set = []
            for j in range(data.size(0)):
                tmp_data, tmp_normal  = self.__farthest_points_normalized(data[j].t(), resample_num, normal[j].t())
                tmp_data_set.append(torch.from_numpy(tmp_data).t().float())
                tmp_normal_set.append(torch.from_numpy(tmp_normal).t().float())
            data = torch.stack(tmp_data_set)
            normal = torch.stack(tmp_normal_set)

        if attack_label == 'Untarget':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        elif attack_label == 'All':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        else:
            assert False

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        if (self.attack_label == 'Untarget'):
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)
            return [pcs, normals, gt_labels]

        else:
            assert False

    def __farthest_points_normalized(self, obj_points, num_points, normal):
        first = np.random.randint(len(obj_points))
        selected = [first]
        dists = np.full(shape = len(obj_points), fill_value = np.inf)

        for _ in range(num_points - 1):
            dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
            selected.append(np.argmax(dists))
        res_points = np.array(obj_points[selected])
        res_normal = np.array(normal[selected])

        # normalize the points and faces
        avg = np.average(res_points, axis = 0)
        res_points = res_points - avg[np.newaxis, :]
        dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
        res_points = res_points / dists

        return res_points, res_normal