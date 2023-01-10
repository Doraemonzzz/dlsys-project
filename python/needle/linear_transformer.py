import math
from typing import List

import numpy as np

import needle.init as init
from needle import ops
from needle.autograd import Tensor
from .nn import *

# 3D version of LinearLayer
class Linear3D(Module):
    def __init__(self, in_features, out_features, device=None, dtype="float32"):
        super().__init__()
        self.linear = Linear(in_features, out_features, device=device, dtype=dtype)
        
    def forward(self, x):
        b, n, d = x.shape
        # b, n, d -> b * n, d
        x = ops.reshape(x, (b * n, d))
        # b * n, d -> b * n, e
        x = self.linear(x)
        # b * n, e -> b, n, e
        d = x.shape[-1]
        x = ops.reshape(x, (b, n, d))
        
        return x

class LinearAttention(Module):
    def __init__(self, d, h, device=None, dtype="float32"):
        super().__init__()
        self.qkv = Linear3D(d, 3 * d, device=device, dtype=dtype)
        self.out = Linear3D(d, d, device=device, dtype=dtype)
        self.d = d
        self.h = h
        self.e = self.d // self.h
        self.act = ReLU()
    
    def forward(self, x, eps=1e-5):
        # b, n, d -> b, n, 3 * d
        qkv = self.qkv(x)
        # get shape
        b = qkv.shape[0]
        n = qkv.shape[1]
        # reshape
        qkv = ops.reshape(qkv, (b, n, 3, self.d))
        # split
        q, k, v = ops.split(qkv, axis=2)
        # b, n, d -> b, n, h, e
        q, k, v = [ops.reshape(x, (b, n, self.h, self.e)) for x in (q, k, v)]
        # b, n, h, e -> b, h, n, e
        q, k, v = [ops.transpose(x, (2, 1)) for x in (q, k, v)]
        # b, h, n, e -> b * h, n, e
        q, k, v = [ops.reshape(x, (b * self.h, n, self.e)) for x in (q, k, v)]
        # act
        q = self.act(q) + eps
        k = self.act(k) + eps
        # (b * h, n, e), (b * h, n, e) -> (b * h, e, e)
        kv = ops.matmul(ops.transpose(k, (2, 1)), v)
        # (b * h, n, e), (b * h, e, e) -> (b * h, n, e)
        output = ops.matmul(q, kv)
        # qk denom
        # 1, n, 1 -> b * h, n, 1
        ones = init.ones(n, device=output.device, dtype=output.dtype)
        ones = ops.broadcast_to(ops.reshape(ones, (1, n, 1)), (k.shape[0], n, 1))
        # (b * h, n, e), (b * h, n, 1) -> (b * h, e, 1)
        t1 = ops.matmul(ops.transpose(k, (2, 1)), ones)
        # (b * h, n, e), (b * h, e, 1) -> (b * h, n, 1)
        t2 = ops.matmul(q, t1)
        # (b * h, n, e), (b * h, n, 1) -> (b * h, n, e)
        output = ops.divide(output, ops.broadcast_to(t2, output.shape))
        # (b * h, n, e) -> (b, h, n, e)
        output = ops.reshape(output, (b, self.h, n, self.e))
        # (b, h, n, e) -> (b, n, h, e)
        output = ops.transpose(output, (2, 1))
        # (b, n, h, e) -> (b, n, d)
        output = ops.reshape(output, (b, n, self.d))
        # (b, n, d) -> (b, n, d)
        output = self.out(output)
        
        return output
    
class FFN(Module):
    def __init__(self, d, device=None, dtype="float32"):
        super().__init__()
        self.module = Sequential(
            Linear3D(d, 2 * d, device=device, dtype=dtype),
            ReLU(),
            Linear3D(2 * d, d, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.module(x)
    
class LinearTransformer(Module):
    def __init__(self, d, h, device=None, dtype="float32"):
        super().__init__()
        self.module = Sequential(
            Residual(
                Sequential(
                    LayerNorm1d(d, device=device, dtype=dtype),
                    LinearAttention(d, h, device=device, dtype=dtype),
                )
            ),
            Residual(
                Sequential(
                    LayerNorm1d(d, device=device, dtype=dtype),
                    FFN(d, device=device, dtype=dtype),
                )
            ),
        )
        
    def forward(self, x):
        return self.module(x)
    
class Reshape(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # b, n, m, d -> b, (n * m), d
        b, n, m, d = x.shape
        return ops.reshape(x, (b, n * m, d))
    
class Mean(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # b, n, d -> b, d
        n = x.shape[1]
        output = ops.divide_scalar(ops.summation(x, 1), n)
        
        return output
    
class PatchEmbedding(Module):
    def __init__(self, d_in, d_out, device=None, dtype="float32"):
        super().__init__()
        # !!! current does not spot kernel size = stride size, will cause pad < 0
        self.conv = Conv(d_in, d_out, 8, 7, device=device, dtype=dtype)
        self.norm = LayerNorm1d(d_out, device=device, dtype=dtype)
        self.reshape = Reshape()
        
    def forward(self, x):
        # b, d, h, w -> b, d1, h1, w1
        x = self.conv(x)
        # b, d1, h1, w1 -> b, w1, h1, d1
        x = ops.transpose(x, (3, 1))
        # b, w1, h1, d1 -> b, h1, w1, d1
        x = ops.transpose(x, (2, 1))
        # b, h1, w1, d1 -> b, h1 * w1, d1
        x = self.reshape(x)
        # b, h1, w1, d1 -> b, h1, w1, d1
        x = self.norm(x)
        
        return x

# class PatchEmbedding(Module):
#     def __init__(self, d_in, d_out, n, device=None, dtype="float32"):
#         super().__init__()
#         self.n = n
#         self.norm = LayerNorm1d(d_out, device=device, dtype=dtype)
#         self.reshape = Reshape()
        
#     def forward(self, x):
#         # b, d, h, w -> b, d1, h1
#         x = ops.reshape
#         # b, d1, h1, w1 -> b, w1, h1, d1
#         x = ops.transpose(x, (3, 1))
#         # b, w1, h1, d1 -> b, h1, w1, d1
#         x = ops.transpose(x, (2, 1))
#         # b, h1, w1, d1 -> b, h1 * w1, d1
#         x = self.reshape(x)
#         # b, h1, w1, d1 -> b, h1, w1, d1
#         x = self.norm(x)
        
        # return x