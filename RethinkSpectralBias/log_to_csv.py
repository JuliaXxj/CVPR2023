import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import argparse

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, '{}'.format(dname))).Reload() for dname in range(65)]
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def tags_to_csv(dpath):
    dirs = os.listdir(dpath)
    dirs = [d for d in dirs if '.csv' not in d]

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def event_to_csv(dpath):
    files = os.listdir(dpath)
    dname = [a for a in files if "events.out.tfevents" in a]
    event_path = os.path.join(dpath, dname[0])
    event = EventAccumulator(event_path).Reload()
    tags = event.scalars.Keys()

    for index, tag in enumerate(tags):
        steps = [e.step for e in event.Scalars(tag)]
        value = [e.value for e in event.Scalars(tag)]
        df = pd.DataFrame({'steps': steps, 'values': value})
        df.to_csv(os.path.join(dpath, "{}.csv".format(tag)))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    # args = sys.argv
    # data_name = args[1]  # 'svhn', 'cifar10', 'cifar100'
    # model_name = args[2]  # 'resnet18', 'resnet18', 'vgg16', 'vgg13', 'vgg11'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', type=str,
                        choices=['resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11', 'fcn', 'cnn', 'nf_resnet18',
                                 'nf_resnet34', 'nf_resnet101'], required=True, help="choose model")
    parser.add_argument('-d', '--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100', 'mnist', 'fmnist'],
                        required=True, help="choose dataset")

    parser.add_argument('-b', '--bias', action="store_true", help="Model with bias or without bias")

    args = parser.parse_args()

    data_name = args.dataset
    model_name = args.model
    bias = args.bias

    runs = 'runs'
    if bias:
        path = os.path.join(runs, data_name, model_name, "bias", 'log')
    else:
        path = os.path.join(runs, data_name, model_name, "nobias", 'log')

    tags_to_csv(os.path.join(path, 'train energy'))
    # path = os.path.join(runs, data_name, model_name, 'log')
    event_to_csv(path)

