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
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
        

class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic=10, omega0=0.1):
        super().__init__()
        self.register_buffer("frequencies", omega0 * (2.0**torch.arange(n_harmonic)),)

    def forward(self, x):
        embed = (x[..., None] * self.frequencies).reshape(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    
class BaseNetwork(nn.Module):
    def __init__(self, n_harmonic=10, omega0=0.1):
        super().__init__()
        self.positional_encoding = HarmonicEmbedding(n_harmonic, omega0)


class PartMLP(BaseNetwork):
    def __init__(self, num_layers=3, input_size=3, output_size=3, hidden_size=64, L=10):
        input_size = L * 2 * input_size
        super().__init__(n_harmonic=L)
        layers = []
        for i in range(num_layers-1):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.001)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.mlp(x)
        return x
    
    
class PrimitiveMLP(BaseNetwork):
    def __init__(self, num_layers=3, input_size=3, output_size=3, hidden_size=128, d_latent=64, L=10):
        input_size = L * 2 * input_size + d_latent
        super().__init__(n_harmonic=L)
        layers = []
        for i in range(num_layers-1):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.001)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, latent_code):
        x = self.positional_encoding(x).repeat(cfg.nb,1,1)
        return self.mlp(torch.cat([x, latent_code[:,None,:].repeat(1,x.shape[1],1)], 2))
