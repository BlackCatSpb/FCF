"""PotentialFunction: V(z): ℝ²⁴ → ℝ — learned scalar potential."""
import torch, torch.nn as nn

class PotentialFunction(nn.Module):
    """Learned scalar potential over coordinate space. Low = frequent region, High = unexplored."""
    
    def __init__(self, dim=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, z):
        return self.net(z).squeeze(-1)
    
    def gradient(self, z):
        z = z.detach().requires_grad_(True)
        v = self(z)
        grad = torch.autograd.grad(v.sum(), z, create_graph=True)[0]
        return grad
    
    def find_minimum(self, z0, steps=50, lr=0.01):
        with torch.enable_grad():
            z = z0.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                loss = self(z).sum()
                loss.backward()
                opt.step()
        return z.detach()
    
    def find_saddle(self, za, zb, n_points=20):
        t = torch.linspace(0, 1, n_points, device=za.device)
        za_flat = za.view(-1); zb_flat = zb.view(-1)
        points = za_flat.unsqueeze(0) + t.unsqueeze(1) * (zb_flat - za_flat).unsqueeze(0)
        vals = self(points)
        idx = vals.argmax()
        return points[idx], vals[idx]
