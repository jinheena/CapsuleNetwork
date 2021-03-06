import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
           
    # x (NCWH) : 128 x 1 x 28 x 28
    # out (NCWH) : 128 x 256 x 20 x 20
    def forward(self,x):
        return F.relu(self.conv(x))

class PrimaryCapsLayer(nn.Module):
    def __init__(self, dim_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) for _ in range(dim_capsules)])

    # x (NCWH) : 128 x 256 x 20 x 20  
    # out : 128 x 1152(=32x6x6) x 8      
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=4)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return  self.squash(u, dim=1)
    
    def squash(self, x, dim):
        x_mag_sq = torch.sum(x**2, dim, keepdim=True)
        x_mag = torch.sqrt(x_mag_sq)
        v_j = (x_mag_sq / (1.0 + x_mag_sq)) * (x / x_mag)
        return v_j

class DigitCapsuleLayer(nn.Module):
    def __init__(self, opt, num_capsules=10, num_routes=32*6*6, num_w=32, in_channels=8, out_channels=16):
        super(DigitCapsuleLayer, self).__init__()
        self.opt = opt
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, num_w, num_capsules, out_channels, in_channels))

    # batch_size = 128
    # u : 128 x 1152(=32x6x6) x 8
    # u stack : 128 x 1152 x 10 x 8 x 1
    # W : 1 x 32 x 10 x 16 x 8  (not 1 x 1152 x 10 x 16 x 8)
    # W_concat : 128 x 1152 x 10 x 16 x 8 (concate twice, dim 1 & dim 0)
    # u_hat(W_concat*u_stack) : 128 x 1152 x 10 x 16 x 1
    def forward(self, u):
        batch_size = u.size(0)
        u_stack = torch.stack([u] * self.num_capsules, dim=2).unsqueeze(4)
        W_concat = torch.cat([self.W] * (6*6), dim=1)
        W_concat = torch.cat([W_concat] * batch_size, dim=0)
        u_hat = torch.matmul(W_concat, u_stack)
        
        # Routing algorithm (Procedure 1 in the paper)
        # b_ij, c_ij : 1 x 1152(=32x6x6) x 10 x 1
        # c_ij_cat : 128 x 1152 x 10 x 1 x 1
        # s_ij (c_ij_cat * u_hat) :  128 x 1152 x 10 x 16 x 1
        # s_j, v_j : 128 x 1 x 10 x 16 x 1
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if self.opt.cuda:
            b_ij = b_ij.cuda()
        num_iterations = 3

        # v_j_cat : 128 x 1152 x 10 x 16 x 1
        # a_ij : 128 x 1152 x 10 x 1 x 1
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            c_ij_cat = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)           
            s_j = (c_ij_cat * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j, dim=3)
            if iteration < num_iterations:
                v_j_cat = torch.cat([v_j] * self.num_routes, dim=1)
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_j_cat) # agreement
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, x, dim):
        x_mag_sq = torch.sum(x**2, dim, keepdim=True)
        x_mag = torch.sqrt(x_mag_sq)
        v_j = (x_mag_sq / (1.0 + x_mag_sq)) * (x / x_mag)
        return v_j

class DecoderNetwork(nn.Module):
    def __init__(self, opt, num_input=16*10, num_output=28*28):
        super(DecoderNetwork, self).__init__()
        self.opt = opt
        self.decoder = nn.Sequential()
        self.decoder.add_module("fc1", nn.Linear(num_input, 512))
        self.decoder.add_module("relu1", nn.ReLU(inplace=True))
        self.decoder.add_module("fc2", nn.Linear(512, 1024))
        self.decoder.add_module("relu2", nn.ReLU(inplace=True))
        self.decoder.add_module("fc3", nn.Linear(1024, num_output))
        self.decoder.add_module("sig1", nn.Sigmoid())


    # x (digit_cap_out) : 128 x 10 x 16 x 1
    # Check : Do i need to use class label for masking?
    def forward(self, x, y=None):
        batch_size = x.size(0)
        class_num = x.size(1)
        
        x_mag= torch.sqrt( (x**2).sum(2) )
        x_mag = F.softmax(x_mag, dim=1)

        max_val, max_idx = x_mag.max(dim=1)
        mask = Variable(torch.sparse.torch.eye(class_num))
        if self.opt.cuda==True:
            mask = mask.cuda()
        
        mask = mask.index_select(dim=0, index=Variable(max_idx.squeeze(1).data))
        if self.opt.recon_with_gt==False:
            decorder_mask = mask
        else:
            decorder_mask = y

        x = (x.squeeze(3) * decorder_mask[:,:,None]).view(batch_size, -1)
        recon = self.decoder(x).view(-1, 1, 28, 28)

        return mask, recon
