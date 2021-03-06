import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.optim import Adam
import cv2
import random

from dataloader import get_mnist_data
from network import CapsuleNetwork
from checkpoint import Checkpoint

_true_set = {'true', '1'}
_false_set = {'false', '0'}
def str2bool(value, raise_exc=True):
    value = value.lower()
    if value in _true_set:
        return True
    if value in _false_set:
        return False
    if raise_exc:
        raise argparse.ArgumentTypeError('Expected "%s"' % '", "'.join(_true_set | _false_set))
        
    return None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist')
parser.add_argument('--batch_size', type=int, help='input batch size', default=128)
parser.add_argument('--n_epochs', type=int, help='number of epoch', default=15)
parser.add_argument('--cuda', type=str2bool, help='enables cuda', default=True)
parser.add_argument('--r_lambda', type=float, help='scale down factor of the reconstruction loss', default=0.0005)
parser.add_argument('--vis', type=str2bool, help='Show reconstructed output', default=False)
parser.add_argument('--recon_with_gt', type=str2bool, help='Use class label for reconstrucntion', default=False)
parser.add_argument('--save_results', type=str2bool, help='Save trained model and images', default=True)
parser.add_argument('--save_folder', help='folder to save output image and model checkpoints', default='./out')
parser.add_argument('--resume', type=str2bool, help='resume training from the previous checkpoint', default=False)
parser.add_argument('--is_train', type=str2bool, help='start training ', default=True)

opt = parser.parse_args()

if opt.save_results:
    try:
        os.mkdir(opt.save_folder)
    except OSError:
        if not os.path.isdir(opt.save_folder):
            raise

def train(epoch, model, train_loader, test_loader, optimizer):
    train_loss = 0
    for batch_id, (data, target) in enumerate(train_loader):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
        if opt.cuda==True:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output, mask, recon = model(data, target)
        loss = model.loss(output, target, recon, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]

        # check accuracy of a batch
        # if batch_id % data.size(0) == 0:
        #     print "epoch : {}, batch_id: {}, train accuracy : {}".format(epoch + 1, batch_id, 
        #         sum(np.argmax(mask.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1)) / float(opt.batch_size) )

        # check error of a batch
        # if batch_id % data.size(0)  == 0 or len(data) < opt.batch_size:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_id * opt.batch_size + data.size(0), len(train_loader.dataset), 100. * (batch_id + 1) / len(train_loader), loss.data[0]))

        if opt.vis==True:
            idx = random.randint(0, data.size(0) - 1)
            show_recon = recon[idx].data.cpu().numpy().reshape(28,28)
            show_data = data[idx].data.cpu().numpy().reshape(28,28)

            cv2.namedWindow("recon", cv2.WINDOW_NORMAL)
            cv2.imshow("recon", np.concatenate( (show_data, show_recon), axis = 1 ) )
            cv2.waitKey(1)

            if batch_id % data.size(0) == 0: # save reconstructed output
                save_name = '%s/recon_%d_%d.png' % (opt.save_folder, epoch, idx)
                cv2.imwrite(save_name, show_recon*255)

    model.eval()
    test_loss = 0
    num_correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
        if opt.cuda==True:
            data, target = data.cuda(), target.cuda()

        output, mask, recon = model(data)
        loss = model.loss(output, target, recon, data)

        test_loss += loss.data[0]
        num_correct = num_correct + sum(np.argmax(mask.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        # check accuracy of a batch
        # if batch_id % data.size(0) == 0 or len(data) < opt.batch_size:
        #     print "epoch: {}, batch_id: {}, test accuracy: {}".format(epoch + 1, batch_id, sum(np.argmax(mask.data.cpu().numpy(), 1) == 
        #                         np.argmax(target.data.cpu().numpy(), 1)) / float(opt.batch_size) )

    print '[Epoch {}] Training / Test Error / Test Accuracy : {:.4f} / {:.4f} / {:.4f} '.format(epoch + 1, 
                                                                    train_loss / len(train_loader), test_loss / len(test_loader),
                                                                    100. * num_correct / len(test_loader.dataset))


def run_test(model, test_loader):
    latest_checkpoint_path = Checkpoint.get_latest_checkpoint(opt.save_folder)
    resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
    model = resume_checkpoint.model
    optimizer = resume_checkpoint.optimizer

    model.eval()
    test_loss = 0
    num_error = 0
    num_data = 0
    for batch_id, (data, target) in enumerate(test_loader):
        data = Variable(data)
        if opt.cuda==True:
            data = data.cuda()

        output, mask, recon = model(data)
        out_mag= torch.sqrt( (output**2).sum(2) )
        out_mag = F.softmax(out_mag, dim=1)
        max_val, max_idx = out_mag.max(dim=1)
        
        for idx in range(data.size(0)):
            print "(batch_index, sample_index, estimated, target) : ", batch_id, idx, max_idx[idx].data.cpu().numpy(), target[idx]
            if max_idx[idx].data.cpu().numpy() != target[idx]:
                num_error = num_error + 1
            num_data = num_data + 1
        if opt.vis==True:
            idx = random.randint(0, data.size(0) - 1)
            show_recon = recon[idx].data.cpu().numpy().reshape(28,28)
            show_data = data[idx].data.cpu().numpy().reshape(28,28)

            cv2.namedWindow("recon", cv2.WINDOW_NORMAL)
            cv2.imshow("recon", np.concatenate( (show_data, show_recon), axis = 1 ) )
            cv2.waitKey(1)
    print 'test error : ', float(num_error) / float(num_data)

def main():
    train_loader, test_loader = get_mnist_data('../%s' %opt.dataset, opt.batch_size)
    model = CapsuleNetwork(opt)
    if opt.cuda==True:
        model = model.cuda()
    
    if opt.is_train==True:
        if opt.resume==True:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(opt.save_folder)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            optimizer = resume_checkpoint.optimizer
            start_epoch = resume_checkpoint.epoch + 1
        else:
            start_epoch = 0
            optimizer = Adam(model.parameters())
        

        for epoch in range(start_epoch, opt.n_epochs):
            train(epoch, model, train_loader, test_loader, optimizer)
            Checkpoint(model=model,optimizer=optimizer, epoch=epoch).save(opt.save_folder)
    else:
        run_test(model, test_loader)

if __name__ == "__main__":
    main()
