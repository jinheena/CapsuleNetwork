import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
import cv2

from dataloader import get_mnist_data
from network import CapsuleNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist')
parser.add_argument('--batch_size', type=int, help='input batch size', default=128)
parser.add_argument('--n_epochs', type=int, help='number of epoch', default=30)
parser.add_argument('--cuda', type=bool, help='enables cuda', default=True)
parser.add_argument('--r_lambda', type=float, help='scale down factor of the reconstruction loss', default=0.0005)
parser.add_argument('--vis', type=bool, help='Show reconstructed output', default=False)
parser.add_argument('--recon_with_gt', type=bool, help='Use class label for reconstrucntion', default=False)

opt = parser.parse_args()

def train(epoch, model, train_loader, test_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_id, (data, target) in enumerate(train_loader):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
        if opt.cuda==True:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        #output = capsule_network(data)
        #loss = capsule_network.margin_loss(output, target)

        output, mask, recon = model(data)
        loss = model.loss(output, target, recon, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]

        if opt.vis==True:
            cv2.namedWindow("recon", cv2.WINDOW_NORMAL)
            show_recon = recon[1].data.cpu().numpy().reshape(28,28)
            show_data = data[1].data.cpu().numpy().reshape(28,28)
            cv2.imshow("recon", np.concatenate( (show_data, show_recon), axis = 1 ) )
            cv2.waitKey(1)

        if batch_id % 100 == 0:
            print "epoch : {}, train accuracy : {}".format(epoch, 
                sum(np.argmax(mask.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1)) / float(opt.batch_size) )
        
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
    print test_loss / len(test_loader)

def main():
    train_loader, test_loader = get_mnist_data('../mnist', opt.batch_size)
    capsule_network = CapsuleNetwork(opt)
    if opt.cuda==True:
        capsule_network = capsule_network.cuda()
    optimizer = Adam(capsule_network.parameters())

    for epoch in range(opt.n_epochs):
        train(epoch, capsule_network, train_loader, test_loader, optimizer)

if __name__ == "__main__":
    main()