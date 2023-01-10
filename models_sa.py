import torch
import numpy as np
import math
import utils_sa
from torchvision import models as torch_models
from torch.nn import DataParallel

class Model:
    def __init__(self, batch_size, gpu_memory):
        self.batch_size = batch_size
        self.gpu_memory = gpu_memory

    def predict(self, x):
        raise NotImplementedError('use ModelTF or ModelPT')

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils_sa.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


class ModelPT(Model):
    """
    Wrapper class around PyTorch models.

    In order to incorporate a new model, one has to ensure that self.model is a callable object that returns logits,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model_name, _model=None, batch_size=None, gpu_memory=None):
        super().__init__(batch_size, gpu_memory)
        if 'cifar100' == model_name:
            self.mean = np.reshape([0.5071, 0.4867, 0.4408], [1, 3, 1, 1])
            self.std = np.reshape([0.2675, 0.2565, 0.2761], [1, 3, 1, 1])
            model = _model.cuda()
            model.float()
        elif 'cifar10' == model_name:
            self.mean = np.reshape([0.4914, 0.4822, 0.4465], [1, 3, 1, 1])
            self.std = np.reshape([0.2023, 0.1994, 0.2010], [1, 3, 1, 1])
            model = _model.cuda()
            model.float()
            
        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)
        model.eval()
        self.model = model

    def predict(self, x):
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)

        with torch.no_grad():  # otherwise consumes too much memory and leads to a slowdown
            x_batch_torch = torch.as_tensor(x, device=torch.device('cuda'))
            logits = self.model(x_batch_torch).cpu().numpy()
        return logits


