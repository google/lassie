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
from config import cfg
from part_vae import *
from model import *


def train_model():        
    print("========== Loading data... ========== ")
    num_imgs, inputs = load_data()

    if not osp.isfile(cfg.vae_model_path):
        print("========== Pre-training Part VAE... ========== ")
        part_vae = PartVAE().to(cfg.device)
        part_vae.train_vae()

    print("========== Initializing LASSIE model... ========== ")
    model = Model(cfg.device, cfg.category, num_imgs=num_imgs)

    print("========== LASSIE optimization... ========== ")
    model.train(inputs)
    model.save_model(osp.join(cfg.model_dir, '%s.pth'%cfg.animal_class))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    args = parser.parse_args()
    cfg.set_args(args.cls)

    if cfg.animal_class in ['horse', 'cow', 'sheep']:
        from pascal_part import *
    else:
        from web_images import *
    
    train_model()
