import argparse

import dataset_organizer
from classifier import Classifier


parser = argparse.ArgumentParser(description='Scene Classifier with CNN')

parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--freeze-layers', default=10, type=int)
parser.add_argument('--learning-rate', default=0.001, type=float)
parser.add_argument('--data-augmentation', default=False, action='store_true')

args = parser.parse_args()


# Parse the data
dataset_organizer.organize()

classifier = Classifier(
    args.learning_rate,
    args.batch_size,
    args.freeze_layers,
    args.data_augmentation,
    '/tmp/b21327694_dataset/train/',
    '/tmp/b21327694_dataset/test/'
)

for epoch in range(0, args.epochs):
    # train for one epoch
    classifier.train()

    # evaluate on test set
    classifier.test()

# Plot the results (average loss, average top1, average top5)
classifier.plot_the_results()
