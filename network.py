from layers import *

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCapsLayer()
        self.digit_capsules = DigitCapsuleLayer()

    def forward(self, x):
        conv_out = self.conv_layer(x)
        primary_caps_out = self.primary_capsules(conv_out)
        return primary_caps_out

    # with max (eq. 4)
    def margin_loss(self, x, y, size_average=True):
        batch_size = x.size(0)
        m_lambda = 0.5

        v_mag = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1)).cuda()
        loss = y*torch.max(zero, 0.9 - v_mag).view(batch_size, -1)**2 + m_lambda * (1.0 - y)*torch.max(zero, v_mag - 0.1).view(batch_size, -1)**2
        return loss.sum(dim=1).mean()