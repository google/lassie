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


import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    AmbientLights,
    PointLights,
    BlendParams,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    Textures,
)
from config import cfg
from data_utils import *


class Renderer():
    def __init__(self, device, mode='text'):
        super().__init__()
        self.device = device
        self.mode = mode
        R, T = look_at_view_transform(dist=5, elev=0, azim=0, device=device)
        # self.cam = PerspectiveCameras(device=device, focal_length=2, R=R, T=T)
        self.cam = OrthographicCameras(device=device, focal_length=1, R=R, T=T)
        self.part_color = torch.zeros(cfg.nb,3).float().to(device)
        for i in range(cfg.nb):
            self.part_color[i,:] = torch.tensor(part_colors[i+1][:3]).to(device)

        if mode == 'sil':
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0,0,0)) 
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=np.log(1./1e-4-1.)*blend_params.sigma,
                faces_per_pixel=50
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=self.cam[0], raster_settings=raster_settings),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            )
            
        elif mode == 'soft':
            self.light = AmbientLights(device=device)
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0,0,0)) 
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=0,
                faces_per_pixel=1,
                bin_size=0,
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=self.cam[0], raster_settings=raster_settings),
                shader=SoftPhongShader(device=device, cameras=self.cam[0], lights=self.light, blend_params=blend_params)
            )

        else:
            if mode == 'part':
                self.light = PointLights(
                    device=device, location=[[0.0, 3.0, 5.0]], ambient_color=((0.6,0.6,0.6),),
                    diffuse_color=((0.2,0.2,0.2),), specular_color=((0.2,0.2,0.2),))
            else:
                self.light = AmbientLights(device=device)
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0,
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=self.cam[0], raster_settings=raster_settings),
                shader=HardPhongShader(device=device, cameras=self.cam[0], lights=self.light)
            )

        self.renderer = renderer.to(device)

    def render(self, verts, faces, verts_color=None, part_idx=-1):
        bs = verts.shape[0]
        if len(verts.shape) == 3:
            nv = verts.shape[1]
            verts_combined = verts
            faces_combined = faces[None,:,:].repeat(bs,1,1)
        else:
            nb, nv = verts.shape[1], verts.shape[2]
            verts_combined = verts.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None,:,:].repeat(bs,1,1)

        if self.mode == 'sil':
            mesh = Meshes(verts=verts_combined, faces=faces_combined)
        else:
            if self.mode == 'part':
                if part_idx == -1:
                    verts_color = self.part_color[None,:,None,:].repeat(bs,1,nv,1)
                else:
                    verts_color = self.part_color[None,part_idx,None,:].repeat(bs,1,nv,1)
            verts_color = verts_color.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
            mesh = Meshes(verts=verts_combined, faces=faces_combined, textures=Textures(verts_rgb=verts_color))

        return self.renderer(mesh)

    def project(self, x):
        return self.cam.transform_points_screen(x, image_size=cfg.input_size)
    
    def set_sigma(self, sigma):
        blend_params = BlendParams(sigma=sigma, gamma=1e-4, background_color=(0,0,0)) 
        self.renderer.shader.blend_params = blend_params
        self.renderer.rasterizer.raster_settings.blur_radius = np.log(1./1e-4-1.)*blend_params.sigma

    def get_part_masks(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        masks = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            fragments = self.renderer.rasterizer(meshes)
            mask = torch.div(fragments.pix_to_face, faces.shape[0], rounding_mode='floor')+1  # (1, H, W, 1)
            mask[fragments.pix_to_face == -1] = 0
            masks.append(mask)
        return torch.cat(masks, 0).permute(0,3,1,2)
    
    def get_verts_vis(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        verts_vis = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            packed_faces = meshes.faces_packed() 
            pix_to_face = self.renderer.rasterizer(meshes).pix_to_face # (1, H, W, 1)
            visible_faces = pix_to_face.unique()
            visible_verts = torch.unique(packed_faces[visible_faces])
            visibility_map = torch.zeros_like(verts_combined[0,:,0])
            visibility_map[visible_verts] = 1
            verts_vis.append(visibility_map.view(nb, nv))
        return torch.stack(verts_vis, 0)
    
    def get_surface_normals(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        normals = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            fragments = self.renderer.rasterizer(meshes)
            vertex_normals = meshes.verts_normals_packed()
            faces_normals = vertex_normals[meshes.faces_packed()]
            ones = torch.ones_like(fragments.bary_coords)
            pixel_normals = interpolate_face_attributes(fragments.pix_to_face, ones, faces_normals)
            normals.append(F.normalize(pixel_normals.mean(3), dim=3))
        return torch.cat(normals, 0).permute(0,3,1,2)
        