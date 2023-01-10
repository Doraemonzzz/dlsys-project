import sys

sys.path.append('./python')
import math

import needle as ndl
import needle.nn as nn
import needle.linear_transformer as lt
import numpy as np

class LinearVit(ndl.nn.Module):
    def __init__(self, d=32, h=2, device=None, dtype="float32"):
        super().__init__()
        self.patch_embedding = lt.PatchEmbedding(3, d, device=device, dtype=dtype)
        self.linear_transformer = nn.Sequential(
            lt.LinearTransformer(d, h, device=device, dtype=dtype),
        )
        self.mean = lt.Mean()
        self.linear = nn.Sequential(
            nn.Linear(d, 128, device=device, dtype=dtype), 
            nn.ReLU(), 
            nn.Linear(128, 10, device=device, dtype=dtype)
        )
        
    def forward(self, x):
        # b, d, h, w -> b, h1 * w1, d1
        x = self.patch_embedding(x)
        # b, h1 * w1, d1 -> b, h1 * w1, d1
        x = self.linear_transformer(x)
        # b, h1 * w1, d1 -> b, d1
        x = self.mean(x)
        # b, d1 -> b, m
        x = self.linear(x)
        
        return x