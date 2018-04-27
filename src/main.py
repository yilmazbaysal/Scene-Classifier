import argparse
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

import dataset_organizer


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')

parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0


def main():
    # Parse the data
    dataset_organizer.organize()

    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = models.vgg16(pretrained=True)

    # model.features = torch.nn.DataParallel(model.features)

    ct = 0
    for name, child in model.named_children():
        ct += 1
        if ct < 1:
            j = 0
            for name2, params in child.named_parameters():
                j += 1
                if j < 7:
                    params.requires_grad = False

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # Change the last layer of the model
    numb = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(numb, 20)])
    model.classifier = nn.Sequential(*features)

    optimizer = torch.optim.SGD(
        model.classifier._modules['6'].parameters(),
        args.lr
    )

    # Data loading code
    train_directory = '/tmp/b21327694_dataset/train/'
    test_directory = '/tmp/b21327694_dataset/test/'

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            train_directory,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            test_directory,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size,
    )

    for epoch in range(0, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer)

        # evaluate on test set
        test(test_loader, model, criterion)


def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inputs, target) in enumerate(train_loader):
        input_var, target_var = Variable(inputs), Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Record loss
        losses.update(loss.data[0], input_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train: [{0}/{1}]\n'
              'Loss: {2} - Avg: {3}\n'.format(
                i + 1,
                len(train_loader),
                losses.val,
                losses.avg,
            )
        )

    print('+++\n')


def test(test_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        # compute output
        input_variable, target_variable = Variable(input), Variable(target)

        outputs = model(input_variable)
        loss = criterion(outputs, target_variable)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, target_variable.data, topk=(1, 5))
        losses.update(loss.data[0], input_variable.size(0))
        top1.update(prec1[0], input_variable.size(0))
        top5.update(prec5[0], input_variable.size(0))

        print('Test: [{0}/{1}]\n'
              'Loss: {2} - Avg: {3}\n'
              'Top1: {4} - Avg: {5}\n'
              'Top5: {6} - Avg: {7}\n'.format(
                i + 1,
                len(test_loader),
                losses.val,
                losses.avg,
                float(top1.val),
                float(top1.avg),
                float(top5.val),
                float(top5.avg)
            )
        )

    print('---\n')

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
