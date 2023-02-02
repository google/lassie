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
import os.path as osp
import numpy as np
import imageio
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, euler_angles_to_matrix
from pytorch3d.utils import ico_sphere
from config import cfg
from data_utils import *
from extractor import *
from clustering import *
from skeleton import *
from part_mlp import *
from part_vae import *
from rendering import *
from losses import *


class Mesh:
    def __init__(self, device, lev=1):
        template_mesh = ico_sphere(lev, device)
        self.faces = template_mesh.faces_list()[0]
        self.verts = template_mesh.verts_list()[0]
        self.nv = self.verts.shape[0]
        self.nf = self.faces.shape[0]
        self.base_shape = self.verts.clone()

class Model(nn.Module):
    def __init__(self, device, category, num_imgs):
        super().__init__()
        self.device = device
        self.num_imgs = num_imgs
        self.extractor = VitExtractor('dino_vits8', device)

        # Define skeleton and part mapping
        self.skeleton = Skeleton(device, category)

        # Rendering
        self.img_std = torch.from_numpy(img_std).float().to(device)
        self.img_mean = torch.from_numpy(img_mean).float().to(device)
        self.sil_renderer = Renderer(device, 'sil')
        self.part_renderer = Renderer(device, 'part')
        self.text_renderer = Renderer(device, 'text')

        # Instance-invariant parameters
        self.feat_verts = torch.zeros(cfg.nb, cfg.d_feat).float().to(device)
        self.rot_id = torch.zeros(3).float().to(device)
        self.bone_rot_rest = nn.Parameter(self.rot_id[None,:].repeat(cfg.nb,1))
        self.bone_scale = nn.Parameter(torch.zeros(cfg.nb).float().to(device))
        self.f_parts = [PartMLP(L=10).to(device) for i in range(cfg.nb)]
        
        self.part_codes = nn.Parameter(torch.zeros(cfg.nb, cfg.d_latent*2).float().to(device))
        part_vae = PartVAE().to(device)
        part_vae.load_model(cfg.vae_model_path)
        self.f_primitive = part_vae.dec
        for p in self.f_primitive.parameters():
            p.requires_grad = False

        # instance-specific parameters
        self.global_scale = nn.Parameter(torch.zeros(num_imgs).float().to(device))
        self.global_trans = nn.Parameter(-self.skeleton.joints[:,:2].mean(0)[None].repeat(num_imgs,1))
        self.global_rot = nn.Parameter(self.rot_id[None,:].repeat(num_imgs,1))
        self.bone_rot = nn.Parameter(self.rot_id[None,None,:].repeat(num_imgs,cfg.nb,1))

        # define mesh vertices
        self.meshes = {
            1: Mesh(device, 1),
            2: Mesh(device, 2),
            3: Mesh(device, 3),
            4: Mesh(device, 4),
        }
        self.nv2lev = {}
        self.verts_sym = {}
        for k in [1,2,3,4]:
            nv, uvs = self.meshes[k].nv, self.meshes[k].verts.clone()
            self.nv2lev[nv] = k
            self.verts_sym[nv] = torch.zeros(cfg.nb*nv).long().to(device)
            verts = self.transform_verts(uvs, self.skeleton.joints[None], use_ellipsoid=True).reshape(cfg.nb*nv,3)
            verts_ref = verts.clone()
            verts_ref[:,0] *= -1
            for i in range(cfg.nb*nv):
                self.verts_sym[nv][i] = ((verts_ref - verts[i,:])**2).sum(1).argmin()

    def get_uvs_and_faces(self, lev=1, gitter=True):
        uvs = self.meshes[lev].verts.clone()
        faces = self.meshes[lev].faces.clone()
        if gitter:
            uvs += torch.randn_like(uvs)*1e-2 / (2**lev)
        return uvs, faces

    def freeze_parts(self, lev=0):
        self.bone_scale.grad[2] = 0
        self.bone_rot.grad[:,:,0] = 0
        self.bone_rot.grad[:,2,:] = 0
        if lev < 3:
            self.bone_scale.grad[self.skeleton.bones_lev[lev]] = 0
            self.bone_rot.grad[:,self.skeleton.bones_lev[lev]] = 0
            self.part_codes.grad[self.skeleton.bones_lev[lev]] = 0

    def init_feat_verts(self, feat_part, part_mapping):
        self.feat_verts = feat_part[part_mapping]

    def update_feat_verts(self, verts_2d, feat_img, saliency):
        with torch.no_grad():
            feat_verts = F.grid_sample(feat_img, verts_2d*2-1, align_corners=True).permute(0,2,3,1)
            sal_verts = F.grid_sample(saliency, verts_2d*2-1, align_corners=True).permute(0,2,3,1)
            feat_verts_avg = (feat_verts * sal_verts).sum(0) / (sal_verts.sum(0) + 1e-6)
            self.feat_verts = 0.9*self.feat_verts + 0.1*feat_verts_avg.mean(1)

    def update_bone_rot_rest(self):
        with torch.no_grad():
            mean_bone_rot = self.bone_rot.mean(0)
            self.bone_rot -= mean_bone_rot[None]
            self.bone_rot_rest += mean_bone_rot

    def symmetrize_verts(self, verts):
        nv = verts.shape[-2]
        verts_sym = self.verts_sym[nv]
        verts = verts.reshape(-1,cfg.nb*nv,3)
        verts_ref = verts.clone()
        verts_ref[:,:,0] *= -1
        return ((verts + verts_ref[:,verts_sym,:])/2).view(-1,cfg.nb,nv,3)

    def transform_verts(self, uvs, joints, joints_rot=None, deform=True, use_ellipsoid=False):
        bs, nv = joints.shape[0], uvs.shape[0]
        joint1 = joints[:,self.skeleton.bones[0],:]
        joint2 = joints[:,self.skeleton.bones[1],:]
        bone_center = (joint1 + joint2)*0.5
        bone_length = ((joint1 - joint2)**2).sum(2).sqrt()*0.5
        if use_ellipsoid:
            verts = uvs[None,:,:].repeat(cfg.nb,1,1)*0.5
        else:
            verts = self.meshes[self.nv2lev[nv]].base_shape[None,:,:].repeat(cfg.nb,1,1)
            codes = self.part_codes[:,:cfg.d_latent] + torch.exp(0.5*self.part_codes[:,cfg.d_latent:])
            verts = self.f_primitive(uvs[None], codes) # nb x nv x 3
            verts /= verts[:,:,1].abs().max(1)[0][:,None,None].detach()
            if deform:
                verts += torch.cat([f(uvs[None]) for f in self.f_parts], 0) # nb x nv x 3

        verts *= bone_length[0,:,None,None]*1.2 # part scaling
        verts = torch.bmm(self.skeleton.bone_rot_init, verts.permute(0,2,1)).permute(0,2,1) # nb x nv x 3
        if deform and not use_ellipsoid:
            verts = self.symmetrize_verts(verts)[0]
        verts = verts[None].repeat(bs,1,1,1).permute(0,1,3,2).view(-1,3,nv) # bs*nb x 3 x nv
        if joints_rot is not None:
            verts = torch.bmm(joints_rot.reshape(-1,3,3), verts) # part rotation
        verts = verts.view(bs,cfg.nb,3,nv).permute(0,1,3,2) + bone_center[:,:,None,:] # part translation
        return verts # bs x nb x nv x 3

    def global_transform(self, joints, verts, rot, trans=None, scale=None):
        bs, nv = verts.shape[0], verts.shape[2]
        if rot.shape[-1] == 6:
            rot_mat = rotation_6d_to_matrix(rot)
        else:
            rot_mat = euler_angles_to_matrix(rot, 'ZXY')
        joints = torch.bmm(rot_mat, joints.permute(0,2,1)).permute(0,2,1)
        verts = torch.bmm(rot_mat, verts.reshape(-1,cfg.nb*nv,3).permute(0,2,1)).permute(0,2,1)
        if trans is not None:
            joints *= scale[:,None,None]*0.1 + 1
            verts *= scale[:,None,None]*0.1 + 1
            joints[...,:2] += trans[:,None,:]
            verts[...,:2] += trans[:,None,:]
        return joints, verts.reshape(-1,cfg.nb,nv,3)

    def get_view(self, joints, verts, view=0):
        bs = joints.shape[0]
        angle = torch.zeros(3).float().to(self.device)
        angle[1] += view * np.pi / 180
        global_rot = matrix_to_rotation_6d(axis_angle_to_matrix(angle))[None,:].repeat(bs,1)
        joints, verts = self.global_transform(joints, verts, global_rot)
        v_min, v_max = verts.min(2)[0].min(1)[0], verts.max(2)[0].max(1)[0]
        center = joints.mean(1)
        scale = ((verts - center[:,None,None,:])**2).sum(3).max(2)[0].max(1)[0].sqrt()
        joints = (joints - center[:,None,:]) / scale[:,None,None]
        verts = (verts - center[:,None,None,:]) / scale[:,None,None,None]
        return joints, verts

    def get_resting_pose(self, uvs, view=0, deform=True):
        bs = 1
        root_rot = self.rot_id[None,None,:].repeat(bs,1,1) # bs x 1 x 3
        rot = torch.cat([root_rot, self.bone_rot_rest[None,:,:]], 1) # bs x nj x 3
        joints_can, joints_rot = self.skeleton.transform_joints(rot, scale=self.bone_scale)
        verts_can = self.transform_verts(uvs, joints_can, joints_rot, deform, False)
        joints, verts = self.get_view(joints_can, verts_can, view)
        return joints, verts

    def forward(self, inputs, uvs, deform=True, use_ellipsoid=False, view=None, stop_at=10):
        bs = self.num_imgs
        root_rot = self.rot_id[None,None,:].repeat(bs,1,1) # bs x 1 x 3
        rot = torch.cat([root_rot, self.bone_rot + self.bone_rot_rest[None,:,:]], 1) # bs x nj x 3
        joints_can, joints_rot = self.skeleton.transform_joints(rot, scale=self.bone_scale)
        verts_can = self.transform_verts(uvs, joints_can, joints_rot, deform, use_ellipsoid)
        if view is not None:
            joints, verts = self.get_view(joints_can, verts_can, view)
        else:
            joints, verts = self.global_transform(joints_can, verts_can, self.global_rot, self.global_trans, self.global_scale)
        verts_combined = verts.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
        verts_2d = self.sil_renderer.project(verts_combined).view(-1,cfg.nb,uvs.shape[0],3)
        joints_2d = self.sil_renderer.project(joints)
        outputs = {}
        outputs['verts'] = verts
        outputs['joints'] = joints
        outputs['verts_can'] = verts_can
        outputs['joints_can'] = joints_can
        outputs['verts_2d'] = verts_2d[...,:2] / cfg.input_size[0]
        outputs['joints_2d'] = joints_2d[...,:2] / cfg.input_size[0]
        return outputs

    def get_texture(self, uvs, images=None, verts=None, verts_2d=None):
        bs, nv = images.shape[0], uvs.shape[0]
        verts_sym = self.verts_sym[nv]
        verts_color = F.grid_sample(images, verts_2d*2-1, align_corners=True).view(bs,3,-1).permute(0,2,1)
        verts_combined = verts.permute(0,3,1,2).view(bs,3,-1).permute(0,2,1)
        visible = verts_combined[:,:,2] >= verts_combined[:,verts_sym,2]
        verts_color = torch.where(visible[:,:,None], verts_color, verts_color[:,verts_sym,:])
        verts_color = verts_color.permute(0,2,1).view(bs, 3, cfg.nb, -1).permute(0,2,3,1)
        return verts_color * self.img_std + self.img_mean

    def calculate_losses(self, inputs, outputs, losses, weights, params):
        uvs, faces = self.get_uvs_and_faces(params['mesh_lev'], True)
                
        for k in weights:
            loss = 0
            if k == 'feat' and weights[k] > 0:
                loss = feature_loss(self.feat_verts, outputs['verts_2d'], inputs['feat_img'], inputs['saliency'])
            elif k == 'sil' and weights[k] > 0:
                loss = sil_loss(outputs['verts'], faces, inputs['part_masks'], self.sil_renderer)
            elif k == 'part_cent' and weights[k] > 0:
                loss = part_center_loss(outputs['joints_2d'], inputs['part_cent'],
                                        self.skeleton.part_mapping, self.skeleton.bones)
            elif k == 'cam_prior' and weights[k] > 0:
                loss = cam_prior_loss(self.global_rot)
            elif k == 'pose_prior' and weights[k] > 0:
                loss = pose_prior_loss(self.bone_scale, self.bone_rot, self.part_codes, 
                                       outputs['joints_can'], self.skeleton.joints_sym)
            elif k == 'shape_prior' and weights[k] > 0:
                loss = shape_prior_loss(self.f_parts, uvs)
            elif k == 'edge' and weights[k] > 0:
                loss = edge_loss(outputs['verts'][:1], faces)
            elif k == 'normal' and weights[k] > 0:
                loss = normal_loss(outputs['verts'][:1], faces)
            elif k == 'laplacian' and weights[k] > 0:
                loss = laplacian_loss(outputs['verts'][:1], faces)

            if k not in losses:
                losses[k] = []
            losses[k].append(loss * weights[k])

    def optimize(self, variables, inputs, params, weights):
        sigma = 5e-3
        uvs, _ = self.get_uvs_and_faces(params['mesh_lev'], True)
        if 'pose_prior' in weights:
            self.update_bone_rot_rest()

        losses = {}
        optimizer = torch.optim.Adam(variables, lr=params['lr'])
        loop = tqdm(range(params['n_iters']))
        for j in loop:
            optimizer.zero_grad()
            outputs = self.forward(inputs, uvs, deform=True)
            self.calculate_losses(inputs, outputs, losses, weights, params)
            loss = sum(losses[k][-1] for k in losses if len(losses[k]) > 0)
            loss.backward()
            self.freeze_parts(params['lev'])
            optimizer.step()
            loop.set_description("Optimization loss: %.4f" % loss.data)
            self.update_feat_verts(outputs['verts_2d'], inputs['feat_img'], inputs['saliency'])

    def train(self, inputs):
        self.skeleton.init_part_mapping(inputs['part_cent'])
        self.init_feat_verts(inputs['feat_part'], self.skeleton.part_mapping)

        print("========== Optimizing global pose... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot]
        params = {'n_iters':100, 'lr':0.05, 'lev':0, 'mesh_lev':2, 'deform':False}
        weights = {'feat':0.5, 'part_cent':2.0, 'cam_prior':0.1}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing hierarchical pose (lev 1)... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot]
        params = {'n_iters':100, 'lr':0.05, 'lev':1, 'mesh_lev':2, 'deform':False}
        weights = {'feat':1.0, 'cam_prior':0.1, 'pose_prior':5e-3}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing hierarchical pose (lev 2)... ========== ")
        params = {'n_iters':100, 'lr':0.05, 'lev':2, 'mesh_lev':2, 'deform':False}
        outputs = self.optimize(var, inputs, params, weights)

        print("========== Optimizing hierarchical pose (lev 3)... ========== ")
        params = {'n_iters':100, 'lr':0.05, 'lev':3, 'mesh_lev':2, 'deform':False}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing base part shape (lev 1) ... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot, self.bone_scale, self.part_codes]
        params = {'n_iters':100, 'lr':0.01, 'lev':1, 'mesh_lev':2, 'deform':False}
        weights = {'feat':1.0, 'cam_prior':0.05, 'pose_prior':5e-3, 'sil':0.05}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing base part shape (lev 2) ... ========== ")
        params = {'n_iters':100, 'lr':0.01, 'lev':2, 'mesh_lev':2, 'deform':False}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing base part shape (lev 3) ... ========== ")
        params = {'n_iters':100, 'lr':0.01, 'lev':3, 'mesh_lev':2, 'deform':False}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing shape... ========== ")
        var = [self.part_codes]
        for i, f in enumerate(self.f_parts):
            var += f.parameters()
        params = {'n_iters':100, 'lr':5e-3, 'lev':3, 'mesh_lev':3, 'deform':True}
        weights = {'feat':1.0, 'pose_prior':5e-3, 'sil':0.1, 'shape_prior':0.1, 
                   'edge':1.0, 'laplacian':1.0, 'normal':0.1}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing all params... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot, self.part_codes]
        for i, f in enumerate(self.f_parts):
            var += f.parameters()
        params = {'n_iters':200, 'lr':5e-3, 'lev':3, 'mesh_lev':3, 'deform':True}
        weights = {'feat':1.0, 'cam_prior':0.05, 'pose_prior':5e-3, 'sil':0.1, 'shape_prior':0.1, 
                   'edge':0.5, 'laplacian':0.5, 'normal':0.05}
        self.optimize(var, inputs, params, weights)

        print("========== Saving results... ========== ")
        with torch.no_grad():
            self.save_results(inputs, params)

    def save_results(self, inputs, params, losses=None):
        uvs, faces = self.get_uvs_and_faces(4, False)
        outputs = self.forward(inputs, uvs, deform=True)
        texture = self.get_texture(uvs, inputs['images'], outputs['verts'], outputs['verts_2d'])
        self.save_pred_masks(inputs['images'], outputs['verts'], faces)
        
        num_imgs = inputs['images'].shape[0]
        for i in range(num_imgs):
            img_part = self.part_renderer.render(outputs['verts'][i:i+1], faces)
            img_text = self.text_renderer.render(outputs['verts'][i:i+1], faces, verts_color=texture[i:i+1])
            cv2.imwrite(osp.join(cfg.output_dir, 'part_%d.png'%i), img2np(img_part, True))
            cv2.imwrite(osp.join(cfg.output_dir, 'text_%d.png'%i), img2np(img_text))
        
        images_part = [[] for i in range(num_imgs)]
        images_text = [[] for i in range(num_imgs)]
        for r in range(37):
            outputs = self.forward(inputs, uvs, deform=True, view=r*10)
            for i in range(num_imgs):
                img_part = self.part_renderer.render(outputs['verts'][i:i+1], faces)
                img_text = self.text_renderer.render(outputs['verts'][i:i+1], faces, verts_color=texture[i:i+1])
                images_part[i].append(img2np(img_part))
                images_text[i].append(img2np(img_text, True))
        for i in range(num_imgs):
            imageio.mimsave(osp.join(cfg.output_dir, 'part_%d.gif'%i), images_part[i])
            imageio.mimsave(osp.join(cfg.output_dir, 'text_%d.gif'%i), images_text[i])

    def save_pred_masks(self, images, verts, faces):
        masks = self.part_renderer.get_part_masks(verts, faces)
        for i in range(images.shape[0]):
            img = (images[i].cpu().permute(1,2,0).numpy() * img_std + img_mean).clip(0,1)
            cmask = part_mask_to_image(masks[i,0].cpu().numpy(), part_colors, img)
            cv2.imwrite(osp.join(cfg.output_dir, 'mask_pred_%d.png'%i), cmask)

    def save_model(self, model_path):
        state_dict = {'main': self.state_dict()}
        for i, f in enumerate(self.f_parts):
            state_dict['f_part_%d'%i] = f.state_dict()
        torch.save(state_dict, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['main'])
        for i, f in enumerate(self.f_parts):
            f.load_state_dict(checkpoint['f_part_%d'%i])
