import numpy as np
import torch

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model, device):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}

    w_l_mean = []
    w_l_var = []
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.to(device)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        # momentum = b / (n + b)

        model(input_var)

        weight_list_mean = []
        weight_list_var = []
        # 获取每轮数值
        for module in momenta.keys():
            # module.momentum = momentum
            weight_list_mean.append(module.running_mean.cpu())
            weight_list_var.append(module.running_var.cpu())
        # 添加到数组
        # mean
        for j in range(len(weight_list_mean)):
            weight_list_mean[j] = weight_list_mean[j].unsqueeze(0)
        if len(w_l_mean) == 0:
            w_l_mean = weight_list_mean.copy()
        else:
            for j in range(len(weight_list_mean)):
                w_l_mean[j] = torch.cat([w_l_mean[j], weight_list_mean[j]], dim=0)

        # var
        for j in range(len(weight_list_var)):
            weight_list_var[j] = weight_list_var[j].unsqueeze(0)
        if len(w_l_var) == 0:
            w_l_var = weight_list_var.copy()
        else:
            for j in range(len(weight_list_var)):
                w_l_var[j] = torch.cat([w_l_var[j], weight_list_var[j]], dim=0)

        n = n + 1

    tmp_mean = []
    tmp_var = []
    for w in w_l_mean:
        tmp_mean.append(torch.from_numpy(np.median(w.view(n, -1), axis=0)).view(w.size()[1:]))
    for w in w_l_var:
        tmp_var.append(torch.from_numpy(np.median(w.view(n, -1), axis=0)).view(w.size()[1:]))
    k = 0
    for module in momenta.keys():
        module.running_mean = tmp_mean[k].to(device)
        module.running_var = tmp_var[k].to(device)
        k = k + 1
    model.apply(lambda module: _set_momenta(module, momenta))