import torch
import torch.nn as nn
import torch.nn.functional as F 

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
    
    def forward(self,x):
        return F.relu(self.conv(x))


class PrimaryCapsLayer(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) for _ in range(num_capsules)])
        print self.capsules
        raw_input(" ")

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)
    
    def squash(self, x):
        x_norm = (x**2).sum(-1, keepdim=True)
        return (x_norm * x) / ( (1.0 + x_norm) * torch.sqrt(x_norm) )

class DigitCapsuleLayer(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32*6*6, in_channels=8, out_channels=16):
        super(DigitCapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, u):
        batch_size = x.size(0)
        u = torch.stack([u] * self.num_capsules, dim=2).unsqueeze(4)        # need to check its meaning
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, u)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, x):
        x_norm = (x**2).sum(-1, keepdim=True)
        return (x_norm * x) / ( (1.0 + x_norm) * torch.sqrt(x_norm) )