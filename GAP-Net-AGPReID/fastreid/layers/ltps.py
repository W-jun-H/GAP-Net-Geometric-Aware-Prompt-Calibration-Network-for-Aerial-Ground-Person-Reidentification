# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TPSGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=128, use_regular_grid=True, grid_size=3, reg_factor=0):
        super(TPSGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.grid_size = grid_size

        # create a grid of control points
        self.control_head, self.inv_L = self._init_tps(grid_size)

    def _init_tps(self, grid_size):
        # Create grid of control points
        # grid_size x grid_size points regular grid [-1, 1]
        axis_coords = np.linspace(-1, 1, grid_size)
        P_y, P_x = np.meshgrid(axis_coords, axis_coords)
        P = np.stack((P_x, P_y), axis=2) # (grid_size, grid_size, 2)
        P = P.reshape(-1, 2) # (N, 2)
        self.P = torch.tensor(P, dtype=torch.float32)
        N = P.shape[0]

        # Compute L matrix
        # Distance squared matrix U
        # U_ij = ||P_i - P_j||^2 * log(||P_i - P_j||^2)
        P_diff = P[:, None, :] - P[None, :, :] # (N, N, 2)
        r_sq = np.sum(P_diff**2, axis=2)
        r_sq[r_sq == 0] = 1 # Avoid log(0)
        U = r_sq * np.log(r_sq)
        np.fill_diagonal(U, 0) # r=0 -> U=0

        # Construct L
        # [K  P]
        # [P^T 0]
        P_aug = np.concatenate((np.ones((N, 1)), P), axis=1) # (N, 3)
        L = np.zeros((N + 3, N + 3))
        L[:N, :N] = U
        L[:N, N:] = P_aug
        L[N:, :N] = P_aug.T
        
        # Add regularization
        if self.reg_factor != 0:
            L[:N, :N] += np.eye(N) * self.reg_factor

        inv_L = np.linalg.inv(L)
        return torch.tensor(P, dtype=torch.float32), torch.tensor(inv_L, dtype=torch.float32)

    def forward(self, theta):
        # theta: (B, N, 2) target control points
        B = theta.size(0)
        device = theta.device
        
        # We need to solve for weights W and affine A
        # [W | A]^T = inv_L * [theta' | 0]^T
        # Target vector Y: [theta_x theta_y] padded with 0s
        # P' = theta
        # Y = [P'; 0] (N+3, 2)
        
        zeros = torch.zeros(B, 3, 2, device=device)
        Y = torch.cat((theta, zeros), 1) # (B, N+3, 2)
        
        # Solve
        # params = inv_L * Y
        inv_L = self.inv_L.to(device).unsqueeze(0).repeat(B, 1, 1) # (B, N+3, N+3)
        params = torch.bmm(inv_L, Y) # (B, N+3, 2)
        
        W = params[:, :self.grid_size**2, :] # (B, N, 2)
        A = params[:, self.grid_size**2:, :] # (B, 3, 2)
        
        # Generate dense grid
        # Create output grid
        grid_h, grid_w = self.out_h, self.out_w
        # create grid in [-1, 1]
        # (H, W, 2)
        # Using meshgrid with indexing 'xy' to match x, y output
        xs = torch.linspace(-1, 1, grid_w, device=device)
        ys = torch.linspace(-1, 1, grid_h, device=device)
        # In newer pytorch meshgrid needs 'indexing' arg, or just use old behavior which is 'ij' then swap
        # Using stack to ensure (W, H)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid = torch.stack((grid_x, grid_y), 2) # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)
        
        # Flatten for transforming
        grid_flat = grid.view(B, -1, 2) # (B, H*W, 2)
        N_points = grid_flat.size(1)
        
        # Compute distances from each grid point to each control point P
        P = self.P.to(device).unsqueeze(0) # (1, N, 2)
        # (B, H*W, 1, 2) - (1, 1, N, 2)
        diff = grid_flat.unsqueeze(2) - P.unsqueeze(1) # (B, H*W, N, 2)
        r_sq = torch.sum(diff**2, 3) # (B, H*W, N)
        r_sq = torch.clamp(r_sq, min=1e-9)
        U = r_sq * torch.log(r_sq) # (B, H*W, N)
        
        # Apply TPS weights
        # f(x) = A [1 x y]^T + Sum(W_i * U(||x - P_i||))
        
        # Affine part
        # [1, x, y]
        ones = torch.ones(B, N_points, 1, device=device)
        # grid_flat is (x, y)
        P_affine = torch.cat((ones, grid_flat), 2) # (B, H*W, 3)
        # A is (B, 3, 2)
        # P_affine * A => (B, H*W, 2)
        grid_affine = torch.bmm(P_affine, A)
        
        # Non-linear part
        # U is (B, H*W, N)
        # W is (B, N, 2)
        # U * W => (B, H*W, 2)
        grid_nonlinear = torch.bmm(U, W)
        
        final_grid = grid_affine + grid_nonlinear
        
        # Reshape to (B, H, W, 2)
        final_grid = final_grid.view(B, grid_h, grid_w, 2)
        
        return final_grid

class LTPS(nn.Module):
    def __init__(self, embed_dim, out_h, out_w, grid_size=3):
        super(LTPS, self).__init__()
        self.grid_size = grid_size
        num_ctrl_pts = grid_size * grid_size
        self.num_ctrl_pts = num_ctrl_pts
        
        # Lightweight regression branch
        # Takes View Token (embed_dim) -> Predicts offsets for control points
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, num_ctrl_pts * 2),
            nn.Tanh() # Limit the output magnitude
        )
        
        # Initialize the regression branch to output identity transformation
        self.reg_branch[2].weight.data.zero_()
        self.reg_branch[2].bias.data.zero_()
        
        # Initial control points (regular grid)
        # Need to match the order in TPSGridGen
        axis_coords = np.linspace(-1, 1, grid_size)
        P_y, P_x = np.meshgrid(axis_coords, axis_coords)
        P = np.stack((P_x, P_y), axis=2).reshape(-1, 2)
        
        # Register initial P as buffer (not trainable param)
        self.register_buffer('initial_P', torch.from_numpy(P).float())
        
        self.tps_grid_gen = TPSGridGen(out_h, out_w, grid_size=grid_size)
        
        # Max deformation scale (0.1 means max shift is 10% of image size)
        self.max_shift = 0.1

    def forward(self, x, view_token):
        """
        x: (B, C, H, W) - Input Patch Feature Map
        view_token: (B, C) - View Token used to predict offsets
        """
        B = x.size(0)
        
        # Predict relative offsets [-1, 1]
        offsets = self.reg_branch(view_token)
        offsets = offsets.view(B, self.num_ctrl_pts, 2)
        
        # Apply scaling and add to initial regular grid
        # theta = initial_P + predicted_offset * scale
        theta = self.initial_P + offsets * self.max_shift
        
        # Generate grid
        grid = self.tps_grid_gen(theta)
        
        # Sample
        # align_corners=False matches most recent implementations
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed
        
        # Sample
        # align_corners=False matches most recent implementations
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed
