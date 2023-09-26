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
from rendering import *

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

    # Rendering 
    i = 2
    rnd = Renderer(cfg.device, 'part') 
    rnd.render(outputs['verts'][i:i+1], faces)

    # Evaluation
    num_pairs = 0
    pck = 0
    for i1 in range(num_imgs):
        for i2 in range(num_imgs):
            if i1 == i2:
                continue
            # Calculate PCK 
            # ...

    print('PCK=%.4f' % pck)

    if cfg.animal_class in ['horse', 'cow', 'sheep']:
        print("========== IOU evaluation... ==========")
        # Calculate IOU
        # ...
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