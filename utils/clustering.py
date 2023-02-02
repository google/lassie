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
import faiss
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from data_utils import *
from config import cfg


def apply_crf(unary, img):
    nc, h, w = unary.shape[:3]
    # initialize CRF
    d = dcrf.DenseCRF2D(w, h, nc)
    d.setUnaryEnergy(unary.reshape(nc,-1))
    # add pairwise potentials
    compat = [50,15]
    img = np.ascontiguousarray((img.permute(1,2,0).cpu().numpy() * img_std + img_mean).clip(0,1)*255.)
    d.addPairwiseGaussian(sxy=3, compat=compat[0], kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=20, srgb=10, rgbim=img.astype(np.uint8), compat=compat[1],
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # inference
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(h,w)

def get_crf_masks(images, masks):
    h, w = images.shape[2], images.shape[3]    
    masks_crf = []    
    for img, mask in zip(images, masks):        
        prob = F.one_hot(mask[0].long(), cfg.nb+1).permute(2,0,1).float() # c x h x w
        prob = F.normalize(prob.clamp(0.5, 1.0), p=1, dim=0).cpu().numpy()
        unary = np.ascontiguousarray(-np.log(prob))        
        mask = apply_crf(unary, img)
        masks_crf.append(torch.from_numpy(mask).to(cfg.device))   
    return torch.stack(masks_crf, 0)[:,None]

def cluster_features(features, saliency_maps, images, masks_fg=None):
    h, w = images[0].shape[2]//8, images[0].shape[3]//8

    # collect salient features
    salient_features = []
    for s, k in zip(saliency_maps, features):
        salient_features.append(k[s.flatten() >= 0.1])
    mean_feat = torch.cat(salient_features, axis=0).mean(0).cpu().numpy()

    # collect foreground features
    fg_features = []
    bg_features = []
    for k in features:
        all_features = np.ascontiguousarray(k.cpu().numpy())
        dist = ((all_features - mean_feat)**2).sum(1)
        dist = (dist / np.max(dist)).flatten()
        fg_features.append(k[dist < 0.8])
        bg_features.append(k[dist >= 0.8])
    fg_features = torch.cat(fg_features, axis=0).cpu().numpy()
    bg_features = torch.cat(bg_features, axis=0).cpu().numpy()
    
    # K-means clustering
    kmeans = faiss.Kmeans(d=fg_features.shape[1], k=cfg.n_clusters, niter=300, nredo=10)
    kmeans.train(fg_features)
    centroids = kmeans.centroids # nc x d
    
    # K-means clustering
    kmeans_bg = faiss.Kmeans(d=bg_features.shape[1], k=cfg.n_clusters, niter=300, nredo=10)
    kmeans_bg.train(bg_features)
    centroids_bg = kmeans_bg.centroids # nc x d
    
    # calculate min distance to cluster centroids
    masks_vit = []
    part_centers = []
    for i, (k, img) in enumerate(zip(features, images)):
        all_features = np.ascontiguousarray(k.cpu().numpy()) # (h*w) x d
        dist, labels = kmeans.index.search(all_features, 1)
        dist_bg, _ = kmeans_bg.index.search(all_features, 1)
        is_fg = dist < dist_bg
        
        # get cluster masks
        if cfg.use_crf:
            dist_to_cent = ((all_features[:,:,None] - centroids.T[None,:,:])**2).sum(axis=1)
            dist_to_cent /= np.max(dist_to_cent)
            unary = np.concatenate((is_fg, dist_to_cent), 1)
            unary = unary.T.reshape(cfg.n_clusters+1, h, w)
            resize_feat = torch.nn.Upsample(size=img.shape[-1], mode='bilinear', align_corners=True)
            unary = resize_feat(torch.from_numpy(unary)[None,...])[0].numpy() # nc x h x w
            mask = apply_crf(unary, img[0])
            if masks_fg is not None:
                mask *= masks_fg[i][0,0].cpu().numpy().astype(np.int64)
        else:
            mask = labels.copy() + 1 
            mask[~is_fg] = 0           
            mask = mask.reshape(h, w)
        masks_vit.append(torch.from_numpy(mask)[None,None,:,:].to(cfg.device).float())

        # calculate cluster centers
        centers = []
        for j in range(cfg.n_clusters):
            center, _, _, _, _ = find_corners(mask == j+1)
            center[:2] /= mask.shape[0]
            centers.append(torch.from_numpy(center)[None,:].float().to(cfg.device))
        part_centers.append(torch.cat(centers, 0))
        
    return masks_vit, part_centers, centroids
