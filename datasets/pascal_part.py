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
import glob
import cv2
import numpy as np
from scipy.io import loadmat
import torch
from config import cfg
from data_utils import *
from extractor import *
from clustering import *


def load_data(phase='train'):
    if phase == 'train' and osp.exists(osp.join(cfg.preprocessed_dir, 'images.npy')):
        inputs = {}
        for np_file in glob.glob(osp.join(cfg.preprocessed_dir, '*.npy')):
            k = np_file.split('/')[-1].split('.')[0]
            inputs[k] = torch.from_numpy(np.load(np_file)).to(cfg.device)
            if k == 'images':
                num_imgs = inputs[k].shape[0]
        inputs['images'] = resize_bilinear(inputs['images'])
        inputs['part_masks'] = resize_nearest(inputs['part_masks'])
        return num_imgs, inputs
    
    print("Reading images and annotations...")
    img_list = osp.join(cfg.pascal_img_set_dir, '%s.txt'%cfg.animal_class)
    with open(img_list, 'r') as f:
        img_files = [cfg.pascal_img_dir + img_file.replace('\n','') for img_file in f.readlines()]
    
    images = []
    masks_gt = []
    bbox_gt = []
    kps_gt = []
    part_centers_gt = []
    
    for i, img in enumerate(img_files):
        img_id = img.split('/')[-1].replace('.jpg','')
        ann_file = osp.join(cfg.pascal_ann_dir, img.split('/')[-1].replace('.jpg', '.mat'))
        ann = loadmat(ann_file)
        obj = ann['anno'][0,0]['objects'][0,0]
        parts = obj["parts"]
        # mask = obj["mask"]

        img = cv2.imread(img)/255.
        part_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        part_centers = np.zeros((16,3))
        keypoints = np.zeros((14,3))

        for j in range(parts.shape[1]):
            part = parts[0,j]
            part_name = part["part_name"][0]
            mask = part["mask"]
            part_idx = part_indices[part_name]
            part_mask[mask > 0] = part_idx

            center, left, right, top, bottom = find_corners(mask)
            part_centers[part_idx-1,:] = center[0], center[1], 1
            if part_name == 'muzzle':
                keypoints[kp_indices[part_name],:] = (center + bottom)/2
            elif part_name == 'tail':
                keypoints[kp_indices[part_name],:] = top
            elif part_name in ['rfuleg', 'rbuleg', 'lfuleg', 'lbuleg']:
                keypoints[kp_indices[part_name],:] = bottom
            elif part_name in ['rflleg', 'rblleg', 'lflleg', 'lblleg']:
                keypoints[kp_indices[part_name],:] = bottom 
            elif part_name in ['leye', 'reye', 'lear', 'rear']:
                keypoints[kp_indices[part_name],:] = center
                
        keypoints[5,2] = 0 # tail top is often occluded
        np.save(osp.join(cfg.preprocessed_dir, 'kps_%s.npy'%img_id), keypoints)
        
        coords_y, coords_x = np.where(part_mask > 0)
        left = np.min(coords_x)
        top = np.min(coords_y)
        width = np.max(coords_x) - left
        height = np.max(coords_y) - top
        bb = process_bbox(left, top, width, height)
        bbox_gt.append(bb)
        
        keypoints[:,0] = ((keypoints[:,0] - bb[0]) / bb[2]) * keypoints[:,2]
        keypoints[:,1] = ((keypoints[:,1] - bb[1]) / bb[3]) * keypoints[:,2]
        part_centers[:,0] = ((part_centers[:,0] - bb[0]) / bb[2]) * part_centers[:,2]
        part_centers[:,1] = ((part_centers[:,1] - bb[1]) / bb[3]) * part_centers[:,2]
        kps_gt.append(torch.tensor(keypoints).float().to(cfg.device))
        part_centers_gt.append(torch.tensor(part_centers).float().to(cfg.device))
        
        img = crop_and_resize(img, bb, cfg.crop_size, rgb=True, norm=True)
        images.append(img)
        if phase == 'eval':
            part_mask = crop_and_resize(part_mask, bb, cfg.crop_size, rgb=False, norm=False)
            masks_gt.append(part_mask)
        
    if phase == 'train':
        print("Extracting DINO features...")
        extractor = VitExtractor('dino_vits8', cfg.device)
        features, saliency = extractor.extract_feat(images)

        print("Clustering DINO features...")
        masks_vit, part_centers, centroids = cluster_features(features, saliency, images)
    
    print("Collecting input batch...")
    inputs = {}
    inputs['kps_gt'] = torch.stack(kps_gt, 0).to(cfg.device)
    inputs['images'] = torch.cat([resize_bilinear(img) for img in images], 0).to(cfg.device)
    
    if phase == 'eval':
        inputs['part_cent'] = torch.stack(part_centers_gt, 0).to(cfg.device)
        inputs['part_masks'] = torch.cat([resize_nearest(m) for m in masks_gt], 0).to(cfg.device)
        inputs['saliency'] = torch.cat([resize_nearest_lr((m>0).float()) for m in masks_gt], 0).to(cfg.device)
        
    else:
        inputs['part_cent'] = torch.stack(part_centers, 0).to(cfg.device)
        inputs['part_masks'] = torch.cat([resize_nearest(m) for m in masks_vit], 0).to(cfg.device)
        inputs['saliency'] = torch.cat([resize_nearest_lr((m>0).float()) for m in masks_vit], 0).to(cfg.device)
        inputs['masks'] = torch.cat([resize_nearest_lr(m) for m in masks_vit], 0).to(cfg.device)

        # reduce feature dimension
        feat_img = torch.stack([k.permute(1,0).view(384,cfg.hw,cfg.hw) for k in features], 0).to(cfg.device)
        feat_sal = feat_img.permute(0,2,3,1)[inputs['saliency'][:,0]>0]
        _, _, V = torch.pca_lowrank(feat_sal, q=cfg.d_feat, center=True, niter=2)
        feat_img = feat_img.permute(1,0,2,3).reshape(384,-1).permute(1,0)
        inputs['feat_img'] = torch.matmul(feat_img, V).permute(1,0).view(cfg.d_feat,-1,cfg.hw,cfg.hw).permute(1,0,2,3)    
        inputs['feat_part'] = torch.matmul(torch.tensor(centroids).to(cfg.device), V)
                
        # save visualizations
        for i in range(len(images)):
            img = (inputs['images'][i].cpu().permute(1,2,0).numpy() * img_std + img_mean).clip(0,1)
            mask = inputs['part_masks'][i].cpu().permute(1,2,0).numpy()[:,:,0]
            cmask = part_mask_to_image(mask, part_colors, img)
            cv2.imwrite(osp.join(cfg.output_dir, 'proc_%d.png'%i), (img*255.).astype(np.uint8))
            cv2.imwrite(osp.join(cfg.output_dir, 'mask_vit_%d.png'%i), cmask)
        
        for k in inputs:
            np.save(osp.join(cfg.preprocessed_dir, k + '.npy'), inputs[k].cpu().numpy())
        
    return len(images), inputs
