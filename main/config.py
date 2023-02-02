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


import sys
import os
import os.path as osp
import torch


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:
    # data
    crop_size = (512, 512)
    input_size = (256, 256)
    hw = crop_size[0]//8
    
    # train settings
    use_crf = False
    n_clusters = 8
    d_latent = 64
    d_feat = 20

    # directory
    curr_dir = osp.dirname(osp.abspath(__file__))
    root_dir = osp.join(curr_dir, '..')
    data_root = osp.join(root_dir, 'data')
    web_img_dir = osp.join(data_root, 'web_images', 'images')
    web_ann_dir = osp.join(data_root, 'web_images', 'annotations')
    pascal_img_dir = osp.join(data_root, 'pascal_part', 'JPEGImages')
    pascal_ann_dir = osp.join(data_root, 'pascal_part', 'Annotations_Part')
    pascal_img_set_dir = osp.join(data_root, 'pascal_part', 'image_sets')
    model_dir = osp.join(root_dir, 'model_dump')
    vae_model_path = osp.join(model_dir, 'primitive_decoder.pth')  
    eval_dir = osp.join(root_dir, 'results', 'eval')
    make_folder(osp.join(data_root, 'preprocessed'))
    make_folder(osp.join(root_dir, 'results'))
    make_folder(eval_dir)

    gpu_id = '0'
    device = torch.device("cuda:%s" % gpu_id)
    print('>>> Using GPU: {}'.format(gpu_id))

    def set_args(self, animal_class):
        self.animal_class = animal_class
        self.category = 'biped' if self.animal_class in ['kangaroo', 'penguin'] else 'quadruped'
        self.nb = 10 if self.category == 'biped' else 16        
        self.input_dir = osp.join(self.web_img_dir, animal_class)
        self.preprocessed_dir = osp.join(self.data_root, 'preprocessed', animal_class)
        self.output_dir = osp.join(self.root_dir, 'results', animal_class)
        make_folder(self.input_dir)
        make_folder(self.preprocessed_dir)
        make_folder(self.output_dir)
        

cfg = Config()

add_pypath(cfg.root_dir)
add_pypath(osp.join(cfg.root_dir, 'networks'))
add_pypath(osp.join(cfg.root_dir, 'datasets'))
add_pypath(osp.join(cfg.root_dir, 'utils'))
