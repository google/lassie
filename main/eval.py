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
from argparse import ArgumentParser
import numpy as np
import torch
from config import cfg
from data_utils import *
from part_vae import *
from model import *


def eval_model():
    print("========== Loading data... ========== ")
    num_imgs, inputs = load_data('eval')
    
    print("========== Preparing LASSIE model... ========== ")
    model = Model(cfg.device, cfg.category, num_imgs=num_imgs)
    model.load_model(osp.join(cfg.model_dir, '%s.pth'%cfg.animal_class))
    rasterizer = model.text_renderer.renderer.rasterizer
    
    print("========== Keypoint transfer evaluation... ========== ")
    uvs, faces = model.get_uvs_and_faces(3, gitter=False)
    outputs = model.forward(inputs, uvs, deform=True)
    
    num_pairs = 0
    pck = 0
    for i1 in range(num_imgs):
        for i2 in range(num_imgs):
            if i1 == i2:
                continue
            kps1 = inputs['kps_gt'][i1].cpu()
            kps2 = inputs['kps_gt'][i2].cpu()
            verts1 = outputs['verts_2d'][i1].cpu().reshape(-1,2)
            verts2 = outputs['verts_2d'][i2].cpu().reshape(-1,2)
            verts1_vis = get_visibility_map(outputs['verts'][i1,None], faces, rasterizer).cpu()
            v_matched = find_nearest_vertex(kps1, verts1, verts1_vis)
            kps_trans = verts2[v_matched]
            valid = (kps1[:,2] > 0) * (kps2[:,2] > 0)
            dist = ((kps_trans - kps2[:,:2])**2).sum(1).sqrt()
            pck += ((dist <= 0.1) * valid).sum() / valid.sum()
            num_pairs += 1
            
    pck /= num_pairs
    print('PCK=%.4f' % pck)
    
    if cfg.animal_class in ['horse', 'cow', 'sheep']:
        print("========== IOU evaluation... ==========")
        iou = 0
        for i in range(num_imgs):
            valid_parts = 0
            masks = get_part_masks(outputs['verts'][i,None], faces, rasterizer).cpu()
            masks_gt = inputs['part_masks'][i,0].cpu()
            iou += mask_iou(masks>0, masks_gt>0)

        iou /= num_imgs
        print('Overall IOU = %.4f' % iou)
        
    with open(osp.join(cfg.output_eval_dir, '%s.txt'%cfg.animal_class) ,'w') as f:
        f.write('PCK = %.4f\n' % pck)
        if cfg.animal_class in ['horse', 'cow', 'sheep']:
            f.write('Overall IOU = %.4f\n' % iou)
        
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    args = parser.parse_args()
    cfg.set_args(args.cls)
    
    if cfg.animal_class in ['horse', 'cow', 'sheep']:
        from pascal_part import *
    else:
        from web_images import *
    
    with torch.no_grad():
        eval_model()
    