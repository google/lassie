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
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from pytorch3d.structures import Meshes
from config import cfg


# Pascal-part to LASSIE parts
part_indices = {
    'head':    2, # head
    'lblleg': 16, # left back lower leg
    'lbuleg': 12, # left back upper leg
    'lear':    2, # left ear
    'leye':    2, # left eye
    'lflleg': 14, # left front lower leg
    'lfuleg': 10, # left front upper leg
    'lhorn':   2, # left horn
    'muzzle':  2, # muzzle
    'neck':    1, # neck
    'rblleg': 15, # right back lower leg
    'rbuleg': 11, # right back upper leg
    'rear':    2, # right ear
    'reye':    2, # right eye
    'rflleg': 13, # right front lower leg
    'rfuleg':  9, # right front upper leg
    'rhorn':   2, # right horn
    'tail':    4, # tail
    'torso':   3, # torso
    'lfho':   14,
    'rfho':   13,
    'lbho':   16,
    'rbho':   15,
    'nose':    2,
    'lfleg':  10,
    'rfleg':   9,
    'lbleg':  12,
    'rbleg':  11,
    'lfpa':   14,
    'rfpa':   13,
    'lbpa':   16,
    'rbpa':   15,
}

# keypoinnt names to indices
kp_indices = {
    'leye':    0,
    'reye':    1,
    'lear':    2,
    'rear':    3,
    'muzzle':  4,
    'tail':    5,
    'rfuleg':  6,
    'rbuleg':  7,
    'lfuleg':  8,
    'lbuleg':  9,
    'rflleg': 10,
    'rblleg': 11,
    'lflleg': 12,
    'lblleg': 13
}

# keypoinnt name mapping
kp_mapping = {
    'Muzzle (tip of nose)':          'muzzle',
    'Right eye':                     'reye',
    'Right ear':                     'rear',
    'Left eye':                      'leye',
    'Left ear':                      'lear',
    'Front right upper leg (knee)':  'rfuleg',
    'Front right lower leg (ankle)': 'rflleg',
    'Front left upper leg (knee)':   'lfuleg',
    'Front left lower leg (ankle)':  'lflleg',
    'Back right upper leg (knee)':   'rbuleg',
    'Back right lower leg (ankle)':  'rblleg',
    'Back left upper leg (knee)':    'lbuleg',
    'Back left lower leg (ankle)':   'lblleg',
    'Tail top':                      'tail'
}

part_colors = np.array([[  0,   0,   0, 0],  # 0
                        [  0,   0,   1, 1],  # 1
                        [  0,   1,   0, 1],  # 2
                        [  1,   0,   0, 1],  # 3
                        [  0,   1,   1, 1],  # 4
                        [  1,   0,   1, 1],  # 5
                        [  1, 0.5,   0, 1],  # 6
                        [  1,   1,   0, 1],  # 7
                        [  1,   0, 0.5, 1],  # 8
                        [  0, 0.5,   1, 1],  # 9
                        [0.5,   1, 0.5, 1],  # 10
                        [  0,   1, 0.5, 1],  # 11
                        [0.5, 0.5,   1, 1],  # 12
                        [  1,   1, 0.5, 1],  # 13
                        [  1, 0.5,   1, 1],  # 14
                        [0.5,   0,   1, 1],  # 15
                        [0.5,   1,   0, 1],  # 16
                        [  1, 0.5, 0.5, 1],  # 17
                        [0.5,   1,   1, 1],  # 18
                        [0.5, 0.5,   1, 1],  # 19
                        [  1,   1, 0.5, 1],  # 20
                        [  1, 0.5,   1, 1],  # 21
                        [0.5,   0,   1, 1],  # 22
                        [0.5,   1,   0, 1],  # 23
                        [  1, 0.5, 0.5, 1],  # 24
                        [0.5,   1,   1, 1]]) # 25

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
norm_transform = transforms.Normalize(img_mean, img_std)
crop_transform = transforms.functional.resized_crop

# resize masks
resize_bilinear = transforms.Resize(cfg.input_size)
resize_nearest = transforms.Resize(cfg.input_size, transforms.InterpolationMode.NEAREST)
resize_nearest_lr = transforms.Resize(cfg.hw, transforms.InterpolationMode.NEAREST)

def crop_and_resize(img, bb, crop_size, rgb=True, norm=True):
    if rgb:
        img = torch.from_numpy(img).permute(2,0,1)[None,:,:,:].to(cfg.device).float()
        img = crop_transform(img, bb[1], bb[0], bb[3], bb[2], crop_size)
        if norm:
            img = norm_transform(img)
    else:
        img = torch.from_numpy(img)[None,None,:,:].to(cfg.device).float()
        img = crop_transform(img, bb[1], bb[0], bb[3], bb[2], crop_size, transforms.InterpolationMode.NEAREST)
    return img

def part_mask_to_image(part_mask, part_colors, img=None):
    output = np.zeros((part_mask.shape[0], part_mask.shape[1], 4))
    for p in range(len(part_colors)):
        output[part_mask == p, :] = part_colors[p]
    if img is not None:
        output[:,:,3] = 1
        output[part_mask==0, :3] = img[part_mask==0, ::-1]
        output[part_mask>0, :3] *= 0.5
        output[part_mask>0, :3] += img[part_mask>0, ::-1]*0.5
    return (output[:,:,2::-1]*255.).astype(np.uint8)

def process_bbox(left, top, width, height, scale=1.5):
    new_size = int(max(width, height)*scale)
    left = (left + left + width)/2 - new_size/2
    top = (top + top + height)/2 - new_size/2
    height, width = new_size, new_size
    return int(left), int(top), int(width), int(height)

def find_corners(mask):
    y, x = np.where(mask > 0)
    if y.shape[0] > 0:
        x_min, x_max = np.argmin(x), np.argmax(x)
        y_min, y_max = np.argmin(y), np.argmax(y)
        center = np.array([x.mean(), y.mean(), 1])
        left = np.array([x[x_min], y[x_min], 1])
        right = np.array([x[x_max], y[x_max], 1])
        top = np.array([x[y_min], y[y_min], 1])
        bottom = np.array([x[y_max], y[y_max], 1])
        return center, left, right, top, bottom
    else:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    
def img2np(img, bgr2rgb=False, overlay=None):
    img_np = img[0].cpu().detach().numpy()
    if bgr2rgb:
        img_np = np.concatenate([img_np[:,:,2::-1], img_np[:,:,3:]], 2)
    if overlay is not None:
        img_overlay = np.zeros_like(img_np)
        img_overlay[:,:,3] = 1
        img_overlay[:,:,:3] = overlay[:,:,:3]
        img_overlay[img_np[:,:,3]>0, :3] = img_np[img_np[:,:,3]>0, :3]
        img_np = img_overlay
    return (img_np*255.).astype(np.uint8)
    
def plot_losses(losses, output_file):    
    plt.figure()
    for k in losses:
        try:
            plt.plot([l.cpu().detach() for l in losses[k]], label=k)
        except:
            plt.plot([l for l in losses[k]], label=k)
    plt.legend()
    plt.savefig(output_file)
    plt.close('all')

def get_part_masks(verts, faces, rasterizer):
    bs, nb, nv = verts.shape[0], verts.shape[1], verts.shape[2]
    verts_combined = verts.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
    faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None,:,:].repeat(bs,1,1)
    meshes = Meshes(verts=verts_combined, faces=faces_combined)
    packed_faces = meshes.faces_packed() 
    packed_verts = meshes.verts_packed()
    fragments = rasterizer(meshes)
    mask = torch.div(fragments.pix_to_face, faces.shape[0], rounding_mode='floor')+1
    mask[fragments.pix_to_face == -1] = 0
    for p in [5,6,7,8]:
        mask[mask == p] = 3
    return mask[0,:,:,0]

def get_visibility_map(verts, faces, rasterizer):
    bs, nb, nv = verts.shape[0], verts.shape[1], verts.shape[2]
    verts_combined = verts.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
    faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None,:,:].repeat(bs,1,1)
    meshes = Meshes(verts=verts_combined, faces=faces_combined)
    packed_faces = meshes.faces_packed() 
    packed_verts = meshes.verts_packed()
    fragments = rasterizer(meshes)
    visible_faces = fragments.pix_to_face.unique()
    visible_verts = packed_faces[visible_faces].unique()
    visibility_map = torch.zeros(packed_verts.shape[0])
    visibility_map[visible_verts] = 1
    return visibility_map

def find_nearest_vertex(keypoints, verts_2d, verts_vis):
    vert_indices = torch.arange(verts_2d.shape[0]).long()
    outputs = torch.zeros(keypoints.shape[0]).long()    
    for i in range(keypoints.shape[0]):
        dist = ((verts_2d[verts_vis > 0] - keypoints[i,:2])**2).sum(1)
        outputs[i] = vert_indices[verts_vis > 0][dist.argmin()]
    return outputs

def mask_iou(m1, m2):
    m1, m2 = m1.float(), m2.float()
    intersection = (m1 * m2).sum()
    union = (m1 + m2).sum() - intersection
    return intersection / union if union > 0 else 0
