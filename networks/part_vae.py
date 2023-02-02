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


import os.path as osp
import cv2
import torch
import torch.nn.functional as F
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tqdm import tqdm
from data_utils import *
from rendering import *
from part_mlp import PrimitiveMLP
from config import cfg


class PartVAE(torch.nn.Module):
    def __init__(self):
        super(PartVAE, self).__init__()
        # encoder
        self.dim = cfg.d_latent
        self.enc1 = torch.nn.Linear(in_features=8, out_features=64)
        self.enc2 = torch.nn.Linear(in_features=64, out_features=256)
        self.enc3 = torch.nn.Linear(in_features=256, out_features=self.dim*2)
        # decoder
        self.dec = PrimitiveMLP(d_latent=cfg.d_latent)

        mesh = ico_sphere(4, cfg.device)
        self.faces = mesh.faces_list()[0]
        self.uvs = mesh.verts_list()[0]
        self.ellipsoid = self.uvs.clone()
        self.cylinder = self.uvs.clone()
        self.cylinder[:,1] = self.uvs[:,1].clip(-0.8,0.8)/0.8
        mids = self.uvs[:,1].abs()<0.8
        radius = (self.uvs[mids,0]**2 + self.uvs[mids,2]**2).sqrt()/0.6
        self.cylinder[mids,0] /= radius
        self.cylinder[mids,2] /= radius
        self.renderer = Renderer(cfg.device, 'part')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x, gitter=True):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x).view(-1, 2, self.dim)
        # reparametrize
        mu = x[:,0,:]
        log_var = x[:,1,:]
        z = self.reparameterize(mu, log_var) # bs x d
        # decoding
        uvs = self.uvs[None].repeat(x.shape[0],1,1)
        x = self.dec(uvs, z, gitter=gitter)
        return x, mu, log_var

    def sample_primitives(self, bs):
        latent_code = torch.rand(bs,8).to(cfg.device)
        shift_x, shift_z = (latent_code[:,0:1]-0.5)*0.2, (latent_code[:,1:2]-0.5)*0.2
        r_top, r_bottom = latent_code[:,2:3]*0.6+0.2, latent_code[:,3:4]*0.6+0.2
        r_x, r_z = latent_code[:,4:5]*0.2+0.7, latent_code[:,5:6]*0.2+0.7
        r_ellipsoid = latent_code[:,6:7]*0.6+0.2
        w_ellipsoid = latent_code[:,7:8,None]*0.8+0.2
        cylinders = self.cylinder.clone()[None].repeat(bs,1,1)
        ellipsoids = self.ellipsoid.clone()[None].repeat(bs,1,1)
        cylinders[:,:,0] *= r_top * (cylinders[:,:,1]+1)/2 + r_bottom * (2-cylinders[:,:,1]-1)/2
        cylinders[:,:,2] *= r_top * (cylinders[:,:,1]+1)/2 + r_bottom * (2-cylinders[:,:,1]-1)/2
        ellipsoids[:,:,0] *= r_ellipsoid
        ellipsoids[:,:,2] *= r_ellipsoid
        primitives = cylinders * (1-w_ellipsoid) + ellipsoids * w_ellipsoid
        primitives[:,:,0] *= r_x
        primitives[:,:,2] *= r_z
        primitives[:,:,0] += shift_x
        primitives[:,:,2] += shift_z
        return latent_code, primitives

    def update_mesh_shape_prior_losses(self, verts, faces, losses):
        mesh = Meshes(verts=verts, faces=faces)
        losses['edge'].append(mesh_edge_loss(mesh) * 0.1)
        losses['normal'].append(mesh_normal_consistency(mesh) * 0.001)
        losses['laplacian'].append(mesh_laplacian_smoothing(mesh, method="uniform") * 0.1)

    def train_vae(self):
        bs = 64
        losses = {'recon':[], 'kld':[], 'edge':[], 'normal':[], 'laplacian':[]}
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loop = tqdm(range(3000))
        for i in loop:
            with torch.no_grad():
                latent_code, primitives = self.sample_primitives(bs)
            optimizer.zero_grad()
            recon, mu, logvar = self(latent_code, gitter=True)
            losses['kld'].append(-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())*0.001)
            losses['recon'].append(((recon - primitives)**2).mean())
            self.update_mesh_shape_prior_losses(recon, self.faces[None].repeat(bs,1,1), losses)
            loss = sum(losses[k][-1] for k in losses if len(losses[k]) > 0)
            loss.backward()
            optimizer.step()
            loop.set_description('Pre-training Part VAE (loss %.4f)' % loss.data)           
        self.visualize_results()
        self.save_model(cfg.vae_model_path)
        return losses

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        
    def visualize_results(self):
        bs = 8
        with torch.no_grad():
            # mean shape
            code_mean = torch.zeros(1,64).to(cfg.device)
            recon1 = self.dec(self.uvs[None], code_mean, gitter=False)
            # randomly sampled input shapes
            code_latent, inputs = self.sample_primitives(bs)
            recon2, mu, logvar = self(code_latent, gitter=False)
            # randomly sampled latent codes
            code_random = torch.randn(8,64).to(cfg.device)
            recon3 = self.dec(self.uvs[None].repeat(bs,1,1), code_random, gitter=False)  
            # render shapes
            recon_mean = self.renderer.render(recon1[:1]*2, self.faces, part_idx=0)
            cv2.imwrite(osp.join(cfg.output_vae_dir, 'mean.png'), img2np(recon_mean))           
            recon_cylinder = self.renderer.render(self.cylinder[None]*2, self.faces, part_idx=0)
            cv2.imwrite(osp.join(cfg.output_vae_dir, 'cylinder.png'), img2np(recon_cylinder))  
            for i in range(bs):
                rendered1 = self.renderer.render(inputs[i:i+1]*2, self.faces, part_idx=0)
                rendered2 = self.renderer.render(recon2[i:i+1]*2, self.faces, part_idx=0)
                rendered3 = self.renderer.render(recon3[i:i+1]*2, self.faces, part_idx=0)
                cv2.imwrite(osp.join(cfg.output_vae_dir, 'input_%d.png'%i), img2np(rendered1))
                cv2.imwrite(osp.join(cfg.output_vae_dir, 'recon_%d.png'%i), img2np(rendered2)) 
                cv2.imwrite(osp.join(cfg.output_vae_dir, 'random_%d.png'%i), img2np(rendered3))   
        