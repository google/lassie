# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix
from config import cfg


class Skeleton():
    def __init__(self, device, category='quadruped'):
        self.device = device
        self.category = category
        if category == 'quadruped':
            joints = [[ 0.0,    0.0,  0.0], # 0:  root
                      [ 0.0,    0.1,  0.6], # 1:  neck
                      [ 0.0,  -0.15,  0.7], # 2:  head
                      [ 0.0,    0.0, -0.8], # 3:  tail top
                      [ 0.0,  -0.15, -1.0], # 4:  tail end
                      [-0.08, -0.15,  0.0], # 5:  upper leg (front right)
                      [ 0.08, -0.15,  0.0], # 6:  upper leg (front left)
                      [-0.08, -0.15, -0.8], # 7:  upper leg (back right)
                      [ 0.08, -0.15, -0.8], # 8:  upper leg (back left)
                      [-0.08, -0.45,  0.0], # 9:  middle leg (front right)
                      [ 0.08, -0.45,  0.0], # 10: middle leg (front left)
                      [-0.08, -0.45, -0.8], # 11: middle leg (back right)
                      [ 0.08, -0.45, -0.8], # 12: middle leg (back left)
                      [-0.08,  -0.7,  0.0], # 13: lower leg (front right)
                      [ 0.08,  -0.7,  0.0], # 14: lower leg (front left)
                      [-0.08,  -0.7, -0.8], # 15: lower leg (back right)
                      [ 0.08,  -0.7, -0.8]] # 16: lower leg (back left)
            self.parent = [0, 0, 1, 0, 3, 0, 0, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12]
            
        else:
            joints = [[ 0.0,  0.0,  0.0], # 0:  root
                      [ 0.0,  0.3,  0.3], # 1:  neck
                      [ 0.0,  0.2,  0.5], # 2:  head
                      [ 0.0, -0.7, -0.3], # 3:  tail top
                      [ 0.0, -1.0, -0.5], # 4:  tail end
                      [-0.2, -0.2,  0.2], # 5:  upper leg (front right)
                      [ 0.2, -0.2,  0.2], # 6:  upper leg (front left)
                      [-0.1, -0.8, -0.2], # 7:  upper leg (back right)
                      [ 0.1, -0.8, -0.2], # 8:  upper leg (back left)
                      [-0.1, -0.9,  0.0], # 9:  lower leg (back right)
                      [ 0.1, -0.9,  0.0]] # 10: lower leg (back left)
            self.parent = [0, 0, 1, 0, 3, 0, 0, 3, 3, 7, 8]
        
        self.joints = torch.tensor(joints).float().to(device)*1.5
        self.nj = len(joints)
        self.nb = self.nj -1
        
        self.euler_convention = ['YZX']*self.nb
        self.visible_parts = torch.arange(self.nb).long().to(device)
        self.bones_lev = {}
        self.bones_lev[0] = torch.arange(self.nb).long().to(device)
        self.bones_lev[1] = torch.tensor([i for i in range(self.nb) if self.parent[i+1] != 0]).long().to(device)
        self.bones_lev[2] = torch.tensor([i for i in range(self.nb) if self.parent[self.parent[i+1]] != 0]).long().to(device)

        self.joint_tree = torch.eye(self.nj).bool().to(device)
        self.bones = torch.zeros(2, self.nb).long().to(device)
        for i in range(1, self.nj):
            j1, j2 = self.parent[i], i
            self.bones[0,i-1] = j1
            self.bones[1,i-1] = j2
            i_back = self.nj - i
            j1, j2 = self.parent[i_back], i_back
            self.joint_tree[j1] = torch.logical_or(self.joint_tree[j1], self.joint_tree[j2])
        
        self.joints_sym = torch.zeros(self.nj).long().to(device)
        for i in range(self.nj):
            joint_flipped = self.joints[i].clone()
            joint_flipped[0] *= -1
            self.joints_sym[i] = ((self.joints - joint_flipped)**2).sum(1).argmin()
            
        self.init_bone_rot()

    def init_bone_rot(self):
        joints1 = self.joints[self.bones[0],:]
        joints2 = self.joints[self.bones[1],:]
        vec_bone = F.normalize(joints2 - joints1, p=2.0, dim=1) # nb x 3
        vec_bone = torch.where(vec_bone[:,1:2]>0, -vec_bone, vec_bone)
        vec_rest = torch.tensor([0,-1,0])[None,:].repeat(self.nb,1).float().to(self.device) # nb x 3
        v = torch.cross(vec_rest, vec_bone, dim=1) # nb x 3
        c = (vec_rest * vec_bone).sum(1)[:,None,None] # nb x 1 x 1
        V = torch.zeros(self.nb,3,3).to(self.device)
        V[:,0,1] += -v[:,2]
        V[:,0,2] += v[:,1]
        V[:,1,0] += v[:,2]
        V[:,1,2] += -v[:,0]
        V[:,2,0] += -v[:,1]
        V[:,2,1] += v[:,0]
        rot = torch.eye(3)[None,:,:].repeat(self.nb,1,1).to(self.device)
        rot += V + torch.bmm(V,V)*(1/(1+c))
        self.bone_rot_init = rot

    def init_part_mapping(self, part_cent):
        part_cent[part_cent[:,:,2]==0][:,:2] = part_cent[part_cent[:,:,2]>0][:,:2].mean(0)
        leg = part_cent[:,:,1].mean(0).argmax()
        dist_leg = ((part_cent[:,:,:1]-part_cent[:,leg,None,:1])**2).sum(2)
        head = dist_leg.mean(0).argmax()
        dist_head = ((part_cent[:,:,:2]-part_cent[:,head,None,:2])**2).sum(2)
        body = torch.minimum(dist_head, dist_leg).mean(0).argmax()
        if self.category == 'quadruped':
            part_mapping = [head, head, body, body, body, body, body, body, leg, leg, leg, leg, leg, leg, leg, leg]
        else:
            part_mapping = [head, head, body, body, body, body, leg, leg, leg, leg]
        self.part_mapping = torch.tensor(part_mapping).long().to(self.device)

    def transform_joints(self, rot, scale=None, symmetrize=False):
        bs = rot.shape[0]
        joints = self.joints.clone()
        # scaling
        if scale is not None:
            scale = (scale + scale[self.joints_sym[1:]-1])*0.5
            for i in range(1, self.nj):
                offset = (joints[i,:] - joints[self.parent[i],:])[None,:] * scale[i-1].clip(-0.5,1.5)
                joints[self.joint_tree[i]] += offset
        # rotation
        results_rot = [torch.eye(3)[None].repeat(bs,1,1).float().to(self.device)]
        results_trans = [joints[None,0].repeat(bs,1)]
        for i in range(1, self.nj):
            rot_mat = euler_angles_to_matrix(rot[:,i,:], self.euler_convention[i-1])
            rot_parent, trans_parent = results_rot[self.parent[i]], results_trans[self.parent[i]]
            joint_rel_trans = (joints[i] - joints[self.parent[i]])[:,None]
            results_rot.append(torch.bmm(rot_parent, rot_mat))
            results_trans.append(torch.matmul(results_rot[i], joint_rel_trans)[:,:,0] + trans_parent)
        joints = torch.stack(results_trans, 1)
        joints_rot = torch.stack(results_rot, 1)[:,1:]
        # symmetrize
        if symmetrize:
            joints = self.symmetrize_joints(joints)
        return joints, joints_rot

    def symmetrize_joints(self, joints):
        joints_ref = joints.clone()
        joints_ref[:,:,0] *= -1
        return (joints + joints_ref[:,self.joints_sym,:])/2
