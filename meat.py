import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from tqdm import tqdm
import numpy as np
import copy

from utils import Logger, save_checkpoint, torch_accuracy, AverageMeter
import update_bn
from attacks import *

parser = argparse.ArgumentParser(description='Median-ensemble Adversarial Training')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--arch', type=str, default="resnet18",
                    help="decide which network to use, choose from resnet18, WRN")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--loss_fn', type=str, default="cent", help="loss function")
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')

parser.add_argument('--me-epoch', type=int, default=75, metavar='N', help='when to ensemble')

parser.add_argument('--resume', type=bool, default=False, help='whether to resume training')
parser.add_argument('--out-dir', type=str, default='./output', help='dir of output')

args = parser.parse_args()

# Training settings
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

weight_decay = 3.5e-3 if args.arch == 'resnet18' else 7e-4
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


class ME(object):
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model = self.model.apply(self.init_weights)
        self.shadow = self.get_model_state()
        self.epoch_weights = []
        self.weight_list = []
        self.backup = {}

    def add_weights(self, weights):
        self.epoch_weights.append(weights)

    def median_integration(self, step):
        for n in range(step):
            if n == 0:
                epoch_weights = [w.unsqueeze(0) for w in self.epoch_weights[n]]
            else:
                w_l = [w.unsqueeze(0) for w in self.epoch_weights[n]]
                for j in range(len(w_l)):
                    if epoch_weights[j].shape[1:] == w_l[j].shape[1:]:
                        epoch_weights[j] = torch.cat([epoch_weights[j], w_l[j]], dim=0)
                    else:
                        print(
                            f"Dimension mismatch at epoch {n}, tensor index {j}: {epoch_weights[j].shape} vs {w_l[j].shape}")

        self.weight_list = epoch_weights

    def middle_2(self, n):
        tmp = []
        for w in self.weight_list:
            tmp.append(torch.from_numpy(np.median(w.view(n, -1), axis=0)).view(w.size()[1:]))

        for idx, (key, value) in enumerate(self.shadow.items()):
            self.shadow[key] = tmp[idx].cuda()

    def apply_middle(self):
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

    def init_weights(self, m):
        with torch.no_grad():
            if type(m) == nn.Conv2d:
                m.weight.fill_(0.0)
            elif type(m) == nn.Linear:
                m.weight.fill_(0.0)
            elif type(m) == nn.BatchNorm2d:
                m.weight.fill_(0.0)


if args.arch == 'resnet18':
    args.lr = 0.01
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                                   [args.lr, args.lr, args.lr / 10, args.lr / 100])[0]
elif args.arch == 'WRN':
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                                   [args.lr, args.lr, args.lr / 10, args.lr / 20])[0]


def train(epoch, model, teacher_model, Attackers, optimizer, device, descrip_str):
    teacher_model.model.eval()

    losses = AverageMeter()
    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()

    pbar = tqdm(train_loader)
    pbar.set_description(descrip_str)

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        x_adv = Attackers.run_specified('PGD_10', model, inputs, target, return_acc=False)

        model.train()
        lr = adjust_learning_rate(epoch)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()

        nat_logit = teacher_model.model(inputs)

        logit = model(x_adv)
        loss = nn.CrossEntropyLoss()(logit, target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        clean_accuracy.update(torch_accuracy(nat_logit, target, (1,))[0].item())
        adv_accuracy.update(torch_accuracy(logit, target, (1,))[0].item())

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    # restore the historical weights
    if epoch >= args.me_epoch:
        current_weights = model.cpu().state_dict()
        current_weights_copy = copy.deepcopy(current_weights)
        teacher_model.add_weights(list(current_weights_copy.values()))


def test(model, teacher_model, Attackers, device):
    model.eval()
    teacher_model.model.eval()

    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()
    me_clean_accuracy = AverageMeter()
    me_adv_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Testing')

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        acc = Attackers.run_specified('NAT', model, inputs, target, return_acc=True)
        adv_acc = Attackers.run_specified('PGD_20', model, inputs, target, category='Madry', return_acc=True)

        me_acc = Attackers.run_specified('NAT', teacher_model.model, inputs, target, return_acc=True)
        me_adv_acc = Attackers.run_specified('PGD_20', teacher_model.model, inputs, target, category='Madry',
                                              return_acc=True)

        clean_accuracy.update(acc[0].item(), inputs.size(0))
        adv_accuracy.update(adv_acc[0].item(), inputs.size(0))
        me_clean_accuracy.update(me_acc[0].item(), inputs.size(0))
        me_adv_accuracy.update(me_adv_acc[0].item(), inputs.size(0))

        pbar_dic['cleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar_dic['me_cleanAcc'] = '{:.2f}'.format(me_clean_accuracy.mean)
        pbar_dic['ema_advAcc'] = '{:.2f}'.format(me_adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return clean_accuracy.mean, adv_accuracy.mean, me_clean_accuracy.mean, me_adv_accuracy.mean


def attack(model, Attackers, device):
    model.eval()

    clean_accuracy = AverageMeter()
    pgd20_accuracy = AverageMeter()
    pgd100_accuracy = AverageMeter()
    mim_accuracy = AverageMeter()
    cw_accuracy = AverageMeter()
    aa_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Attacking all')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, targets = inputs.to(device), targets.to(device)

        acc_dict = Attackers.run_all(model, inputs, targets)

        clean_accuracy.update(acc_dict['NAT'][0].item(), inputs.size(0))
        pgd20_accuracy.update(acc_dict['PGD_20'][0].item(), inputs.size(0))
        pgd100_accuracy.update(acc_dict['PGD_100'][0].item(), inputs.size(0))
        mim_accuracy.update(acc_dict['MIM'][0].item(), inputs.size(0))
        cw_accuracy.update(acc_dict['CW'][0].item(), inputs.size(0))
        aa_accuracy.update(acc_dict['AA'][0].item(), inputs.size(0))

        pbar_dic['clean'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['PGD20'] = '{:.2f}'.format(pgd20_accuracy.mean)
        pbar_dic['PGD100'] = '{:.2f}'.format(pgd100_accuracy.mean)
        pbar_dic['MIM'] = '{:.2f}'.format(mim_accuracy.mean)
        pbar_dic['CW'] = '{:.2f}'.format(cw_accuracy.mean)
        pbar_dic['AA'] = '{:.2f}'.format(aa_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return [clean_accuracy.mean, pgd20_accuracy.mean, pgd100_accuracy.mean, mim_accuracy.mean, cw_accuracy.mean,
            aa_accuracy.mean]


def main():
    best_acc_clean = 0
    best_acc_adv = best_me_acc_adv = 0
    start_epoch = 1

    if args.arch == "resnet18":
        model = ResNet18(num_classes=args.num_classes)
    if args.arch == "preactresnet18":
        model = PreActResNet18(num_classes=args.num_classes)
    if args.arch == "WRN":
        model = Wide_ResNet(depth=34, num_classes=args.num_classes, widen_factor=10, dropRate=0.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    teacher_model = ME(model)
    Attackers = AttackerPolymer(args.epsilon, args.num_steps, args.step_size, args.num_classes, device)
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title='reweight')

    if not args.resume:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)

        logger_test.set_names(['Epoch', 'Natural', 'PGD20', 'me_Natural', 'me_PGD20'])

        step = 1
        for epoch in range(start_epoch, args.epochs + 1):

            descrip_str = 'Training epoch:{}/{}'.format(epoch, args.epochs)

            train(epoch, model, teacher_model, Attackers, optimizer, device, descrip_str)

            if epoch >= args.me_epoch:
                teacher_model.median_integration(step)
                teacher_model.middle_2(n=step)
                teacher_model.apply_middle()
                update_bn.bn_update(train_loader, teacher_model.model.to(device), device)

            nat_acc, pgd20_acc, me_nat_acc, me_pgd20_acc = test(model, teacher_model, Attackers, device=device)

            logger_test.append([epoch, nat_acc, pgd20_acc, me_nat_acc, me_pgd20_acc])

            if pgd20_acc > best_acc_adv:
                print('==> Updating the best model..')
                best_acc_adv = pgd20_acc
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))

            if me_pgd20_acc > best_me_acc_adv:
                print('==> Updating the teacher model..')
                best_me_acc_adv = me_pgd20_acc
                torch.save(teacher_model.model.state_dict(), os.path.join(args.out_dir, 'me_bestpoint.pth.tar'))

    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar')))
    teacher_model.model.load_state_dict(torch.load(os.path.join(args.out_dir, 'me_bestpoint.pth.tar')))
    res_list = attack(model, Attackers, device)
    res_list1 = attack(teacher_model.model, Attackers, device)

    logger_test.set_names(
        ['Epoch', 'clean', 'PGD20', 'PGD100', 'MIM', 'CW', 'AA'])
    logger_test.append(
        [1000000, res_list[0], res_list[1], res_list[2], res_list[3], res_list[4], res_list[5]])
    logger_test.append(
        [1000001, res_list1[0], res_list1[1], res_list1[2], res_list1[3], res_list1[4], res_list1[5]])

    logger_test.close()


if __name__ == '__main__':
    main()
