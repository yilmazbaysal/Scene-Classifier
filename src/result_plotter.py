from matplotlib import pyplot


def plot_the_results(train_results, test_results):
    pyplot.figure(1)

    # Loss
    pyplot.subplot(212)
    pyplot.plot(
        [x['loss'] for x in train_results],
        'co-',
        [x['loss_avg'] for x in train_results],
        'r-'
    )
    pyplot.title('Loss')
    pyplot.legend(['Loss', 'Average Loss'])

    # Top1 - Top5 accuracies
    pyplot.subplot(221)
    pyplot.plot(
        [x['top1'] for x in test_results],
        'go-',
        [x['top1_avg'] for x in test_results],
        'r-'
    )
    pyplot.title('Top1 (Accuracy)')
    pyplot.legend(['Top1', 'Average Top1'])

    pyplot.subplot(222)
    pyplot.plot(
        [x['top5'] for x in test_results],
        'bo-',
        [x['top5_avg'] for x in test_results],
        'r-'
    )
    pyplot.title('Top5 (Accuracy)')
    pyplot.legend(['Top5', 'Average Top5'])

    pyplot.show()


def parse_the_file(file_path):
    train_result_dict_list = []
    test_result_dict_list = []

    with open(file_path) as f:
        for epoch in f.read().split('\n\n---\n\n'):
            if epoch:
                train, test = epoch.split('\n\n+++\n\n')

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


train_results, test_results = parse_the_file('/home/yilmaz/Desktop/results_yilmaz/lr-01_epoch-1_batch-8.txt')


plot_the_results(train_results, test_results)
