import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
# from torch.autograd.gradcheck import zero_gradients
import time
import shutil
import sys
import numpy as np

def pgd_miat1(inputs, net1, net2, norm_std, norm_mean, denorm_std, denorm_mean, targets=None, step_size=2/255, num_steps=20, epsil=8./255, ch=0, n_iter=None):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    input_shape = inputs.shape
    pert_image = inputs.clone().cuda()
    inputs_ori = inputs.clone().cuda()

    scale = 1.0 / norm_std.mean()
    pert_image = pert_image + (torch.rand(input_shape).cuda()-0.5) * 2 * epsil.view(-1, 1, 1, 1) * scale
    inputs_ori /= denorm_std.view(1,3,1,1).cuda()
    inputs_ori -= denorm_mean.view(1,3,1,1).cuda()
    x_idx = torch.from_numpy(np.arange(pert_image.size(0))).cuda()
    sm = nn.Softmax(dim=1).cuda()
    
    m = 0
    v = 0
    t = 1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    for ii in range(num_steps):
        pert_image.requires_grad_()
        if pert_image.grad is not None:
            pert_image.grad.data.zero_()
        fs1 = net1.eval()(pert_image)
        fs1_copy = fs1.clone().detach()
        fs1_copy[x_idx, targets] = -1e19
        _, tar1 = fs1_copy.max(1)
        
        fs2 = net2.eval()(pert_image)
        fs2_copy = fs2.clone().detach()
        fs2_copy[x_idx, targets] = -1e19
        _, tar2 = fs2_copy.max(1)
        
        if ch  == 0:
            loss_wrt_label = nn.CrossEntropyLoss()(fs1, targets) - nn.CrossEntropyLoss()(fs1, tar1) + nn.CrossEntropyLoss()(fs2, targets) - nn.CrossEntropyLoss()(fs2, tar2)
        else:
            loss_wrt_label = nn.CrossEntropyLoss()(fs1, targets) + nn.CrossEntropyLoss()(fs2, targets)
        grad = torch.autograd.grad(loss_wrt_label, pert_image, only_inputs=True, create_graph=True, retain_graph=False)[0]
        grad.detach_()
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        grad = m_hat / (v_hat ** 0.5 + epsilon)
        t += 1
        
        dr = torch.sign(grad.data)
        pert_image.detach_()
        pert_image += dr * step_size * scale
        
        pert_image /= denorm_std.view(1,3,1,1).cuda()
        pert_image -= denorm_mean.view(1,3,1,1).cuda()
        pert_image = torch.clamp(pert_image, 0., 1.)
        
        r_tot = pert_image - inputs_ori
        r_tot = torch.max(torch.min(r_tot, epsil.view(-1,1,1,1)), -epsil.view(-1,1,1,1))
        pert_image = torch.clamp(inputs_ori + r_tot, 0., 1.)
        pert_image -= norm_mean.view(1,3,1,1).cuda()
        pert_image /= norm_std.view(1,3,1,1).cuda()

    return pert_image.detach()
    