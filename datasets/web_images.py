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
import csv
import json
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
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
    images = []
    bbox_gt = []
    kps_gt = []
    
    with open(osp.join(cfg.web_ann_dir, '%s.csv'%cfg.animal_class), 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_id = str(row['id'])
            img_file = osp.join(cfg.web_img_dir, '%s/input_%s.png'%(cfg.animal_class, img_id))
            if not osp.isfile(img_file):
                r = requests.get(row['img_url'], allow_redirects=True)
                open(img_file, 'wb').write(r.content)
                    
            try:
                img = cv2.imread(img_file)/255.
            except:
                continue
                    
            keypoints = np.zeros((14, 3))
            if 'kps' in row:
                kps = json.loads(row['kps'])
                for kp in kps:
                    kp_name = kp['keypointlabels'][0]
                    kp_id = kp_indices[kp_mapping[kp_name]]
                    keypoints[kp_id,:] = kp['x']*img.shape[1]/100, kp['y']*img.shape[0]/100, 1            
                left = np.min(keypoints[keypoints[:,2]>0, 0])
                top = np.min(keypoints[keypoints[:,2]>0, 1])
                width = np.max(keypoints[keypoints[:,2]>0, 0]) - left
                height = np.max(keypoints[keypoints[:,2]>0, 1]) - top            
            else:
                left, top, width, height = 0, 0, img.shape[1], img.shape[0]

            bb = process_bbox(left, top, width, height)
            bbox_gt.append(bb)

            keypoints[:,0] = ((keypoints[:,0] - bb[0]) / bb[2]) * keypoints[:,2]
            keypoints[:,1] = ((keypoints[:,1] - bb[1]) / bb[3]) * keypoints[:,2]
            kps_gt.append(torch.tensor(keypoints).float().to(cfg.device))

            img = crop_and_resize(img, bb, cfg.crop_size, rgb=True, norm=True)
            images.append(img)
    
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
    
    if phase == 'train':
        inputs['part_cent'] = torch.stack(part_centers, 0).to(cfg.device)
        inputs['part_masks'] = torch.cat([resize_nearest(m) for m in masks_vit], 0).to(cfg.device)
        inputs['saliency'] = torch.cat([resize_nearest_lr((m>0).float()) for m in masks_vit], 0).to(cfg.device)
        inputs['masks'] = torch.cat([resize_nearest_lr(m) for m in masks_vit], 0).to(cfg.device)
        
        # Reduce feature dimension
        feat_img = torch.stack([k.permute(1,0).view(384,cfg.hw,cfg.hw) for k in features], 0).to(cfg.device)
        feat_sal = feat_img.permute(0,2,3,1)[inputs['saliency'][:,0]>0]
        _, _, V = torch.pca_lowrank(feat_sal, q=cfg.d_feat, center=True, niter=2)
        feat_img = feat_img.permute(1,0,2,3).reshape(384,-1).permute(1,0)
        inputs['feat_img'] = torch.matmul(feat_img, V).permute(1,0).view(cfg.d_feat,-1,cfg.hw,cfg.hw).permute(1,0,2,3)
        inputs['feat_part'] = torch.matmul(torch.tensor(centroids).to(cfg.device), V)
        
        for i in range(len(images)):
            img = (inputs['images'][i].cpu().permute(1,2,0).numpy() * img_std + img_mean).clip(0,1)
            mask = inputs['part_masks'][i].cpu().permute(1,2,0).numpy()[:,:,0]
            cmask = part_mask_to_image(mask, part_colors, img)
            cv2.imwrite(osp.join(cfg.output_dir, 'proc_%d.png'%i), (img*255.).astype(np.uint8))
            cv2.imwrite(osp.join(cfg.output_dir, 'mask_vit_%d.png'%i), cmask)
        
        for k in inputs:
            np.save(osp.join(cfg.preprocessed_dir, k + '.npy'), inputs[k].cpu().numpy())
            
    return len(images), inputs
