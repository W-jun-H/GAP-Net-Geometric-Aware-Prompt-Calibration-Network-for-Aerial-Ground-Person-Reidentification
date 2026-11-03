# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import trunc_normal_

class DAM(nn.Module):
    def __init__(self, channel, num_prototypes=32):
        super(DAM, self).__init__()
        self.channel = channel
        self.num_prototypes = num_prototypes
        
        # Class-specific (or rather, category-specific semantic) prototypes
        # initialized as learnable parameters
        self.prototypes = nn.Parameter(torch.zeros(1, num_prototypes, channel))
        trunc_normal_(self.prototypes, std=.02)
        
        self.scale = channel ** -0.5
        
        # Generation of channel-level visibility masks
        # Maps the similarity-to-prototypes distribution (K,) to channel masks (C,)
        self.mask_gen = nn.Sequential(
            nn.Linear(num_prototypes, channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel),
            nn.Sigmoid()
        )
        
        # Initialize mask_gen to output values close to 1.0
        # 1. Zero out weights/biases for the first linear layer
        self.mask_gen[0].weight.data.zero_()
        self.mask_gen[0].bias.data.zero_()
        # 2. Zero output weights, but set positive bias for the final linear layer
        self.mask_gen[2].weight.data.zero_()
        self.mask_gen[2].bias.data.fill_(4.0) # Sigmoid(4.0) ~= 0.98

    def forward(self, x):
        """
        x: (B, N, C) feature maps
        """
        B, N, C = x.shape
        
        # Prototypes as Query, Patch Features as Key
        # But for 'score per patch', we measure similarity of each patch to all prototypes.
        
        # Normalized similarity seems more robust for 'matching'
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        
        # Calculate similarity [B, N, K]
        # x: [B, N, C], prototypes: [1, K, C]
        sim = torch.matmul(x_norm, proto_norm.transpose(1, 2))
        
        # Generate visibility mask based on how well patches match the prototypes
        # [B, N, K] -> [B, N, C]
        mask = self.mask_gen(sim)
        
        # Apply mask
        out = x * mask
        
        return out, mask
