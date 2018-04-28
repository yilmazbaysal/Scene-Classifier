import argparse

import dataset_organizer
from classifier import Classifier


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


# Parse the data
dataset_organizer.organize()

args = parser.parse_args()

classifier = Classifier(args.lr, args.batch_size, '/tmp/b21327694_dataset/train/', '/tmp/b21327694_dataset/test/')

for epoch in range(0, args.epochs):
    # train for one epoch
    classifier.train()

    # evaluate on test set
    classifier.test()

# Plot the results (average loss, average top1, average top5)
classifier.plot_the_results()