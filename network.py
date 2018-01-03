from layers import *

class CapsuleNetwork(nn.Module):
    def __init__(self, opt):
        super(CapsuleNetwork, self).__init__()
        self.opt = opt
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCapsLayer()
        self.digit_capsules = DigitCapsuleLayer(opt)
        self.mse_loss = nn.MSELoss()
        

    def forward(self, x):
        #print 'x : {}'.format(x.shape)
        conv_out = self.conv_layer(x)
        #print 'conv_out : {}'.format(conv_out.shape)
        primary_caps_out = self.primary_capsules(conv_out)
        #print 'primary_caps_out : {}'.format(primary_caps_out.shape)
        digit_caps_out = self.digit_capsules(primary_caps_out)
        
        return digit_caps_out

    # Eq. 4
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
        self.mse_loss = (x.view(x.size(0), -1), x_gt.view(x_gt.size(0), -1))
        r_lambda = 0.0005
        return r_lambda*self.mse_loss

    def loss(self, y_pred, y_gt, x_pred, x_gt):
        return self.margin_loss(pred_y, y_gt) + self.reconstruct_loss(x_pred, x_gt)
