import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

def L2Norm(r):
    r_reshaped = r.view(r.shape[0], -1, *(1 for _ in range(r.dim() - 2)))
    r /= torch.norm(r_reshaped, dim=1, keepdim=True) + 1e-10
    return r

'''def L2Norm(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1, 1)
    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    # print(torch.norm(d.view(d.size(0), -1), dim=1))
    return d'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''def L2Norm(d):

    d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d).to(device)'''

'''def getAdvImage(x,r_adv):
    figure, ax = plt.subplots(1, 2, figsize=(32, 32))
    ax[0].imshow(x.squeeze(0).cuda().detach().cpu())
    ax[0].set_title('Clean Example', fontsize=20)
    ax[1].imshow(x)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].imshow(r_adv.squeeze(0).cuda().detach().cpu())
    ax[1].set_title('Adversarial Example', fontsize=20)
    plt.show()'''

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        model.eval()
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor from gaussian distribution
        r = torch.randn(x.shape).sub(0.5).to(x.device)
        r = L2Norm(r)

        for _ in range(self.vat_iter):
            r.requires_grad_()
            advExamples = x + self.xi * r
            advPredictions = F.log_softmax(model(advExamples), dim=1)  #Log to eliminate the negative KL divergence
            adv_distance = F.kl_div(advPredictions, pred, reduction='batchmean')
            adv_distance.backward()
            r = L2Norm(r.grad)
            model.zero_grad()

        # calc loss
        r_adv = r * self.eps
        advImage = x + r_adv
        advExamples = model(advImage)
        advPredictions = F.log_softmax(advExamples, dim=1)
        loss = F.kl_div(advPredictions, pred, reduction='batchmean')

        '''writer = SummaryWriter()
        grid_x = torchvision.utils.make_grid(x)
        writer.add_image('image', grid_x, 0)
        grid = torchvision.utils.make_grid(advImage)
        writer.add_image('perturbation', grid, 0 + 1)'''

        #figure, ax = plt.subplots(1, 2, figsize=(32, 32))
        #ax[0].imshow(x.squeeze(0).cuda().detach().cpu())
        #ax[0].set_title('Clean Example', fontsize=20)
        # ax[1].imshow(r_adv)
        # ax[1].set_title('Perturbation', fontsize=20)
        #ax[1].imshow(advImage.squeeze(0).cuda().detach().cpu())
        #ax[1].set_title('Adversarial Example', fontsize=20)
        #plt.show()

        return loss