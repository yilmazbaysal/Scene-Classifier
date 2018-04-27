import re

from matplotlib import pyplot



def plot_the_results(train_results, test_results):
    pyplot.figure(1)

    pyplot.subplot(311)
    pyplot.plot(
        [x['loss_avg'] for x in train_results],
        'co-'
    )

    pyplot.subplot(312)
    pyplot.plot(
        [x['top1_avg'] for x in test_results],
        'ro-'
    )

    pyplot.subplot(313)
    pyplot.plot(
        [x['top5_avg'] for x in test_results],
        'bo-'
    )

    pyplot.show()


def parse_the_file(file_path):
    train_result_dict_list = []
    test_result_dict_list = []

    with open(file_path) as f:
        for epoch in f.read().split('\n\n\n---\n\n\n'):
            if epoch:
                train, test = epoch.split('\n\n\n')

                # Parse the training metrics
                train = train.split('\n\n')
                for i in range(len(train)):
                    if train[i]:
                        batch_resuts = train[i].splitlines()

                        loss = batch_resuts[1].split()

                        train_result_dict_list.append(
                            {
                                'loss': float(loss[1]),
                                'loss_avg': float(loss[4])
                            }
                        )

                # Parse the test metrics
                test = test.split('\n\n')
                for i in range(len(test)):
                    if test[i]:
                        batch_resuts = test[i].splitlines()

                        loss = batch_resuts[1].split()
                        top1 = batch_resuts[2].split()
                        top5 = batch_resuts[3].split()

                        test_result_dict_list.append(
                            {
                                'loss': float(loss[1]),
                                'loss_avg': float(loss[4]),
                                'top1': float(top1[1]),
                                'top1_avg': float(top1[4]),
                                'top5': float(top5[1]),
                                'top5_avg': float(top5[4]),
                            }
                        )

    return train_result_dict_list, test_result_dict_list


train_results, test_results = parse_the_file('/home/yilmaz/Desktop/results/lr-0001_epoch-5_batch-8.txt')


plot_the_results(train_results, test_results)
