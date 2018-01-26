# A PyTorch implementation of Capsule Network
A Pytorch implementation of [Dynamic Routing Between Capsules]. / (https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf) 
## How to run Training
```
$ python main.py --dataset mnist
```
## How to run Testing
```
$ python main.py --dataset mnist --is_train False
```
## How to run Training from checkpoint
```
& python main.py --dataset mnist --is_train True --resume=True
```
# Author
Jin Hee Na / jinheena@gmail.com