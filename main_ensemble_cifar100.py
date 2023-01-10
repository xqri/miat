'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
import numpy as np
from utils_pgd_ori import *
from utils_miat_pgd1 import *
from utils_miat_pgd2 import *

# ## SA
from attack import *
import models_sa
import utils_sa

## CW
from attack_cw import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.004, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc2 = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

norm_std = torch.Tensor([0.2675, 0.2565, 0.2761])
norm_mean = torch.Tensor([0.5071, 0.4867, 0.4408])
denorm_std = torch.Tensor([1/0.2675, 1/0.2565, 1/0.2761])
denorm_mean = torch.Tensor([-0.5071, -0.4867, -0.4408])

# Data
bs = 8
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),#(0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),#
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),#(0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),#
])
root = './'
E = 80
trainset = torchvision.datasets.CIFAR100(
    root= root + '', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=512 + E, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(
    root= root + '', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=500, shuffle=False, num_workers=4)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net1 = ResNet_ensemble(depth=20, leaky_relu=False, num_classes=100).cuda()
net2 = ResNet_ensemble(depth=20, leaky_relu=False, num_classes=100).cuda()


#checkpoint = torch.load('***.pth') 
#net1.load_state_dict(checkpoint['net1'])
#net2.load_state_dict(checkpoint['net2'])
#best_acc = checkpoint['acc']
#best_acc2 = checkpoint['acc2']
print('best:', best_acc, best_acc2)

linear0 = nn.Linear(200, 100).cuda()
#checkpoint = torch.load('lp0.pth')
#linear0.load_state_dict(checkpoint['linear'])

#linear = nn.Linear(200, 100).cuda()
#checkpoint = torch.load('lp.pth')
#linear.load_state_dict(checkpoint['linear'])
#linear.eval()
    
net_ori = ResNet18(100).cuda()
#checkpoint_ori = torch.load("ta.pth")
#net_ori.load_state_dict(checkpoint_ori["net"])、

criterion = nn.CrossEntropyLoss()
criterion_adv = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction='sum')

optimizer_ori = optim.SGD(net_ori.parameters(), lr=0.02,
                      momentum=0.9, weight_decay=5e-4)
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer0 = optim.SGD(linear0.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)

sm = nn.Softmax(dim=1).cuda()
saveid = 0
now_id = 0
model_id = 0
for i in range(2000):
        
    train_loss = 0.0
    train_loss_2 = 0.0
    correct = 0
    correct2 = 0
    correct3 = 0
    correct0 = 0
    total_train0 = 0
    total_train = 0
    total_train2 = 0
    total_train3 = 0
    iterations = [100, 50]
    steps = [4./255, 2./255, 1./255]
    epses = [8./255, 4./255, 2./255, 1./255]
    
    ALL = 512
    K = 4
    sz_h = (ALL - bs*K) // 12 // bs
    left = sz_h * 2 #((ALL - bs*8) // 4 + bs*8) // bs
    right = sz_h * 2 + K
    for n_iter, (image, label) in enumerate(trainloader):
        label = label[:bs*(right + sz_h*10)+E].cuda()
        image = image[:bs*(right + sz_h*10)+E]
        eps = torch.tensor(np.random.choice([1/255, 2/255, 4/255, 8/255], bs*sz_h*2)).cuda().float()

        step_size = np.random.choice(steps, 1)[0]
        if n_iter % 2 == 1:
            image_per1 = pgd_ori(image[:bs*sz_h].cuda(), net1, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[:bs*sz_h], step_size=step_size, num_steps=20, epsil=eps[:bs*sz_h], ch=0)
            image_per2 = pgd_ori(image[bs*sz_h:bs*left].cuda(), net2, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[bs*sz_h:bs*left], step_size=step_size, num_steps=20, epsil=eps[bs*sz_h:], ch=0)
        else:
            image_per1 = pgd_ori(image[:bs*sz_h].cuda(), net1, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[:bs*sz_h], step_size=step_size, num_steps=20, epsil=eps[:bs*sz_h], ch=1)
            image_per2 = pgd_ori(image[bs*sz_h:bs*left].cuda(), net2, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[bs*sz_h:bs*left], step_size=step_size, num_steps=20, epsil=eps[bs*sz_h:], ch=1)

        
        net_sa1 = models_sa.ModelPT(model_name='cifar100', _model=net1)
        net_sa2 = models_sa.ModelPT(model_name='cifar100', _model=net2)
        
        image_per6 = (image[bs*(right+sz_h*6):bs*(right+sz_h*8)] / denorm_std.view(1,3,1,1)) - denorm_mean.view(1,3,1,1)
        image_per6 = image_per6.numpy()
        label6 = label[bs*(right+sz_h*6):bs*(right+sz_h*8)].cpu().numpy()
        eps = 8/255
        y_target_onehot = utils_sa.dense_to_onehot(label6, n_cls=100)
        if n_iter % 2 == 1:
            n_queries, image_per6 = square_attack_linf2(net_sa1, net_sa2, image_per6, y_target_onehot, eps=eps, n_iters=800,
                                    p_init=0.8, targeted=False, loss_type='margin_loss')
        else:
            n_queries, image_per6 = square_attack_linf_max(net_sa1, net_sa2, image_per6, y_target_onehot, eps=eps, n_iters=800,
                                    p_init=0.8, targeted=False, loss_type='margin_loss')


        image_per6 = torch.from_numpy(image_per6).cuda().float()
        image_per6 = (image_per6 - norm_mean.view(1,3,1,1).cuda()) / norm_std.view(1,3,1,1).cuda()
        
 
        image_per7 = (image[bs*(right+sz_h*8):bs*(right+sz_h*10)] / denorm_std.view(1,3,1,1)) - denorm_mean.view(1,3,1,1)
        image_per7 = torch.clamp(image_per7, 0., 1.)
        if n_iter % 3 == 0:
            image_per7 = cw_gen(net1, net2, image_per7, label[bs*(right+sz_h*8):bs*(right+sz_h*10)], max_iterations=30, norm_std=norm_std, norm_mean=norm_mean, num_cls=100, ch=0, linear=linear0)
        elif n_iter % 3 == 1:
            image_per7 = cw_gen(net1, net2, image_per7, label[bs*(right+sz_h*8):bs*(right+sz_h*10)], max_iterations=30, norm_std=norm_std, norm_mean=norm_mean, num_cls=100, ch=1, linear=linear0)
        else:
            image_per7 = cw_gen(net1, net2, image_per7, label[bs*(right+sz_h*8):bs*(right+sz_h*10)], max_iterations=30, norm_std=norm_std, norm_mean=norm_mean, num_cls=100, ch=2, linear=linear0)


        eps = torch.tensor(np.random.choice([8/255], image[bs*right:bs*(right+sz_h*6)].size(0))).cuda().float()

        sz = eps.size(0) // 3
        if n_iter % 2 == 1:
            image_per3 = pgd_miat1(image[bs*right:bs*right+sz].cuda(), net1, net2, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[bs*right:bs*right+sz], step_size=step_size, num_steps=40, epsil=eps[:sz], n_iter=n_iter, ch=0)
        else:
            image_per3 = pgd_miat1(image[bs*right:bs*right+sz].cuda(), net1, net2, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[bs*right:bs*right+sz], step_size=step_size, num_steps=40, epsil=eps[:sz], n_iter=n_iter, ch=1)

        image_per4 = pgd_miat2(image[bs*right+sz:bs*right+sz*2].cuda(), net1, net2, linear0, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[bs*right+sz:bs*right+sz*2], step_size=step_size, num_steps=20, epsil=eps[sz:sz*2], ch=0)

        eps = torch.tensor(np.random.choice([8/255], bs*sz_h*2)).cuda().float()

        image_per5 = pgd_miat2(image[bs*right+sz*2:bs*right+sz*3].cuda(), net1, net2, None, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[bs*right+sz*2:bs*right+sz*3], step_size=step_size, num_steps=20, epsil=eps, ch=1)
        
        
        eps = torch.tensor(np.random.choice([8/255], E)).cuda().float()
        image_per8 = pgd_ori(image[-E:].cuda(), net_ori, norm_std, norm_mean, denorm_std, denorm_mean, targets=label[-E:], step_size=step_size, num_steps=20, epsil=eps, ch=1)

        
        image_per3 = torch.cat((image[bs*left:bs*right].cuda(), image_per3, image_per4, image_per5, image_per6, image_per7, image_per8))
        size3 = image_per3.size(0)
        
        net1.train()
        net2.train()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        image_per = torch.cat((image_per1, image_per2, image_per3)).detach()
        del image_per1
        del image_per2
        del image_per3
        del image_per4
        del image_per5
        del image_per6
        del image_per7
        #del image_per8

        x_idx = torch.from_numpy(np.arange(image_per.size(0))).cuda()
        outputs1 = net1(image_per)
        outputs2 = net2(image_per)
        outs0 = torch.cat((outputs1, outputs2), dim=-1)
        outs0.detach_()
        
        outs_sm1 = sm(outputs1)
        outs_sm2 = sm(outputs2)
        outs_tar1 = outs_sm1[x_idx, label]
        outs_tar2 = outs_sm2[x_idx, label]
        outs_sm_copy1 = outs_sm1.detach().clone()
        outs_sm_copy1[x_idx, label] = 0
        outs_sm_copy2 = outs_sm2.detach().clone()
        outs_sm_copy2[x_idx, label] = 0
        
        outs_tar = torch.cat((outs_tar1.view(-1, 1), outs_tar2.view(-1, 1)), dim=-1)
        _, idx_max_tar = outs_tar.max(1)
        outs_max_tar = outs_tar[x_idx, idx_max_tar]
        
        outs_sm_copy = torch.cat((outs_sm_copy1, outs_sm_copy2), dim=-1)
        _, idx_max_rest = outs_sm_copy.max(1)
        idx_max_rest = idx_max_rest.detach()
        outs_max_rest = outs_sm_copy[x_idx, idx_max_rest]
        mask = (outs_max_rest > outs_max_tar * 1.0) 
        mask[bs*left:bs*right] = (outs_max_rest[bs*left:bs*right] > outs_max_tar[bs*left:bs*right] * 0.85)
        mask = mask.detach()
        
        weights = torch.ones(mask.size(0)).cuda() * 5
        weights[bs*right:bs*right+sz] = 15
        weights[bs*right+sz:bs*right+sz*2] = 15
        weights[bs*right+sz*2:bs*right+sz*3] = 15
        weights[bs*right+sz*3:-E] = 15
        weights[-E:] = 10
        
        outs = torch.cat((outs_sm1, outs_sm2), dim=-1)
        if mask.sum().item() > 0:
            loss_r = (-torch.log(1 - torch.clip(outs[x_idx, idx_max_rest][mask], 0.001, 0.999)) * weights[mask]).mean()
        else:
            loss_r = torch.tensor([0]).cuda()
        
        sel1 = mask[-size3:]
        sel12 = mask[-size3:]
        sel2 = outs_tar1[-size3:] > outs_tar2[-size3:]
        sel3 = mask[bs*sz_h:bs*left]
        sel4 = mask[:bs*sz_h]
        
        b1_mask = torch.cat((torch.zeros(bs*sz_h).bool().cuda(),torch.ones(bs*sz_h).bool().cuda() & sel3.detach(), sel2.detach() & sel1.detach()))
        b2_mask = torch.cat((torch.ones(bs*sz_h).bool().cuda() & sel4.detach(), torch.zeros(bs*sz_h).bool().cuda(), (~sel2.detach()) & sel12.detach()))
        
        loss_adv1 = 0
        loss_adv2 = 0
        if b1_mask.sum().item() > 0:
            loss_adv1 = criterion(outputs1[b1_mask], label[b1_mask])
        if b2_mask.sum().item() > 0:
            loss_adv2 = criterion(outputs2[b2_mask], label[b2_mask])
        loss_adv = loss_adv1 + loss_adv2
        loss = loss_adv + loss_r
        
        _, preds = outs.max(1)
        preds = preds % 100
        total_train += preds[bs*right+sz:bs*right+sz*3].size(0)
        total_train2 += bs*sz_h*4
        correct += preds[bs*right+sz:bs*right+sz*3].eq(label[bs*right+sz:bs*right+sz*3]).sum().item()
        correct2 += preds[bs*(right+sz_h*6):bs*(right+sz_h*10)].eq(label[bs*(right+sz_h*6):bs*(right+sz_h*10)]).sum().item()
        
        if loss > 0:
            train_loss += loss.item()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
       
        linear0.train()
        optimizer0.zero_grad()
        outs0 = linear0(outs0)
        loss = criterion(outs0, label)
        _, preds = outs0.max(1)
        correct0 += preds.eq(label).sum().item()
        total_train0 += label.size(0)
        loss.backward()
        optimizer0.step()
       
        net_ori.train()
        optimizer_ori.zero_grad()
        outs_ori = net_ori(image_per8[-E:].cuda())
        loss = criterion(outs_ori, label[-E:])
        _, preds = outs_ori.max(1)
        loss.backward()
        optimizer_ori.step()
 
        acc = correct / total_train
        acc2 = correct2 / total_train2
        acc0 = correct0 / total_train0
            
        if (n_iter+1) % 10 == 0:
            print('boundary:', mask[:bs*left].sum().item(), mask[bs*left:bs*right].sum().item(), mask[bs*right:bs*right+sz].sum().item(), mask[bs*right+sz:bs*right+sz*2].sum().item(), mask[bs*right+sz*2:bs*right+sz*3].sum().item(), mask[bs*(right+sz_h*6):bs*(right+sz_h*8)].sum().item(), mask[bs*(right+sz_h*8):bs*(right+sz_h*10)].sum().item(), mask[-E:].sum().item())

            print(acc, acc2, acc0, loss_adv.item(), loss_r.item())
            print('%d/%d | Loss: %.3f | Acc: %.4f%% %.4f%%' % (n_iter+1, len(trainloader), train_loss/(n_iter+1), 100.*acc, 100*acc2))
            net1.eval()
            net2.eval()
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            for _, (image, label) in enumerate(testloader):
                image, label = image.cuda(), label.cuda()
                with torch.no_grad():
                    outs = torch.cat((sm(net1(image)), sm(net2(image))), dim=-1)
                _, preds = outs.max(1)
                preds = preds % 100
                correct_test += preds.eq(label).sum()
                total_test += len(label)
                
            acc_test = correct_test * 1.0 / total_test
            print('Test acc', acc_test)
            if acc_test > best_acc2:
                print('save2')
                best_acc2 = acc_test
                state = {
                    'net1': net1.state_dict(),
                    'net2': net2.state_dict(),
                    'acc': best_acc,
                    'acc2': best_acc2,
                }
                #torch.save(state, '***_best.pth')
                
            
        if (n_iter + 1) % 10 == 0:
            print('save')
            state = {
                'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'acc': best_acc,
                'acc2': best_acc2,
            }
            #torch.save(state, '***_overfit.pth')

            state = {
                    'linear': linear0.state_dict(),
                    'acc': best_acc,
                }
            #torch.save(state, 'lp0.pth')
       
            state = {
                    'net': net_ori.state_dict(),
                    'acc': best_acc,
                }
            #torch.save(state, 'ta.pth')
     
        if (n_iter + 1) == 50:
            print('break')
            break
        
    print('ACC:', best_acc, best_acc2)
    

