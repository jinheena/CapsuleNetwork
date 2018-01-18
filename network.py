from layers import *

class CapsuleNetwork(nn.Module):
    def __init__(self, opt):
        super(CapsuleNetwork, self).__init__()
        self.opt = opt
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCapsLayer()
        self.digit_capsules = DigitCapsuleLayer(opt)
        self.decoder = DecoderNetwork(opt)
        self.r_lambda = opt.r_lambda

    def forward(self, x, y=None):
        conv_out = self.conv_layer(x)
        primary_caps_out = self.primary_capsules(conv_out)
        digit_caps_out = self.digit_capsules(primary_caps_out)
        mask, recon = self.decoder(digit_caps_out, y)
        
        return digit_caps_out, mask, recon
    
    def mse_loss(self, input, target):
        return torch.sum((input - target)**2) / input.data.nelement() 

    def margin_loss(self, y, y_gt, size_average=True):
        batch_size = y.size(0)
        v_mag = torch.sqrt((y**2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        if self.opt.cuda==True:
            zero, y_gt = zero.cuda(), y_gt.cuda()
        m_plus = 0.9
        m_minus = 0.1
        m_lambda = 0.5

        pos_max = torch.max(zero,  m_plus - v_mag).view(batch_size, -1).view(batch_size, -1)**2
        neg_max = torch.max(zero, v_mag - m_minus).view(batch_size, -1).view(batch_size, -1)**2
        loss = y_gt * pos_max + m_lambda * (1.0 - y_gt) * neg_max
        return loss.sum(dim=1).mean()

    def reconstruct_loss(self, x, x_gt):        
        return self.r_lambda * self.mse_loss(x.view(x.size(0), -1), x_gt.view(x_gt.size(0), -1))

    def loss(self, y, y_gt, x, x_gt):
        return self.margin_loss(y, y_gt) + self.reconstruct_loss(x, x_gt)
