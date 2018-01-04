import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms

from dataloader import get_mnist_data
from network import CapsuleNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist')
parser.add_argument('--batch_size', type=int, help='input batch size', default=128)
parser.add_argument('--n_epochs', type=int, help='number of epoch', default = 30)
parser.add_argument('--cuda', type=bool, help='enables cuda', default = True)
opt = parser.parse_args()
print opt

def main():
   
    train_loader, test_loader = get_mnist_data('../mnist', opt.batch_size)

    capsule_network = CapsuleNetwork(opt)
    if opt.cuda==True:
        capsule_network = capsule_network.cuda()
    optimizer = Adam(capsule_network.parameters())
    
    for epoch in range(opt.n_epochs):
        print 'epoch {}'.format(epoch)
        capsule_network.train()
        train_loss = 0
        for batch_id, (data, target) in enumerate(train_loader):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)
            if opt.cuda==True:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            #output = capsule_network(data)
            #loss = capsule_network.margin_loss(output, target)

            output, mask, recon = capsule_network(data)
            loss = capsule_network.loss(output, target, recon, data)
            loss.backward()

            optimizer.step()
            train_loss += loss.data[0]
            
            if batch_id % 100 == 0:
                print "train accuracy:", sum(np.argmax(mask.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(opt.batch_size)
            
        print train_loss / len(train_loader)
            
        capsule_network.eval()
        test_loss = 0
        for batch_id, (data, target) in enumerate(test_loader):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)
            if opt.cuda==True:
                data, target = data.cuda(), target.cuda()

            output, mask, recon = capsule_network(data)
            loss = capsule_network.loss(output, target, recon, data)

            test_loss += loss.data[0]
            
            if batch_id % 100 == 0:
                print "test accuracy:", sum(np.argmax(mask.data.cpu().numpy(), 1) == 
                                    np.argmax(target.data.cpu().numpy(), 1)) / float(opt.batch_size)
        
        print test_loss / len(mnist.test_loader)


if __name__ == "__main__":
    main()