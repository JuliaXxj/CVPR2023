from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
import numpy as np
import random
import sys
import os

from NFnets.nfnets import models
from NFnets.nfnets.models import resnet as nfresnet

from NFnets.nfnets.sgd_agc import SGD_AGC
from NFnets.nfnets.agc import AGC

# from FCN.models import MNISTNet, FeedforwardNeuralNetModel

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model', type=str, choices=['resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11', 'fcn', 'nf_resnet18', 'nf_resnet34'], required=True, help="choose model")
parser.add_argument('-d', '--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100', 'mnist'], required=True, help="choose dataset")
parser.add_argument('-d', '--data_path', type=str, default='./data', required=True, help="path for save data")
parser.add_argument('-b', '--bias', type=bool, action="store_true", help="Model with bias or without bias")
# parser.add_argument('--load_model_path', type=str, help="Model with bias or without bias")



parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--train_bs', type=int, default=128, help="training batch size")
parser.add_argument('--epoch', type=int, default=20, help="training epoch")
parser.add_argument('--eval_bs', type=int, default=250, help="eval batch size")
parser.add_argument('--label_noise', type=float, default=0.1, help="label noise")
parser.add_argument('--delta_h', type=float, default=0.5)
parser.add_argument('--nb_interpolation', type=128, default=128)

# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = sys.argv
# data_name = args[1]     # 'svhn', 'cifar10', 'cifar100'
# data_root = args[2]
# model_name = args[3]    # 'resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11'

# setting
# lr = 1e-4
# train_batch_size = 128
# train_epoch = 1000
# eval_batch_size = 250
# label_noise = 0.10
# delta_h = 0.5
# nb_interpolation = 128

args = parser.parse_args()

data_name = args.dataset
data_root = args.data_path
model_name = args.model
bias = args.bias
load_model = False
# if args.load_model is not None:
#     load_model = True
#     load_model_path = args.load_model

lr = args.lr
train_batch_size = args.train_bs
train_epoch = args.epoch
eval_batch_size = args.eval_bs
label_noise = args.label_noise
delta_h = args.delta_h
nb_interpolation = args.nb_interpolation


if data_name == 'cifar10':
    dataset = datasets.CIFAR10
    num_classes = 10
    from archs.cifar10 import vgg, resnet
elif data_name == 'cifar100':
    dataset = datasets.CIFAR100
    num_classes = 100
    from archs.cifar100 import vgg, resnet
elif data_name == 'svhn':
    dataset = datasets.SVHN
    num_classes = 10
    from archs.svhn import vgg, resnet
elif data_name == 'mnist':
    dataset = datasets.MNIST
    num_classes = 10
    from FCN.models import MNISTNet, FeedforwardNeuralNetModel

else:
    raise Exception('No such dataset')

if model_name == 'vgg11':
    model = vgg.vgg11_bn()
elif model_name == 'vgg13':
    model = vgg.vgg13_bn()
elif model_name == 'vgg16':
    model = vgg.vgg16_bn()
elif model_name == 'resnet18':
    model = resnet.resnet18()
elif model_name == 'resnet34':
    model = resnet.resnet34()
elif model_name.startswith("nf"):
    from NFnets.nfnets.models import resnet as nfresnet
    model = nfresnet.__dict__[model_name](num_classes=num_classes, bias=bias)
elif model_name == 'fcn':
    model = FeedforwardNeuralNetModel(28*28, 128, num_classes, bias=bias)
else:
    raise Exception("No such model!")




if data_name in ['svhn', 'cifar10', 'cifar100']:
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    eval_transform = transforms.Compose([transforms.ToTensor()])
elif data_name in ['mnist']:
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    eval_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
else:
    raise Exception('No such dataset for transformation!')

# load data
if 'cifar' in data_name or 'mnist' in data_name:
    train_data = dataset(data_root, train=True, transform=train_transform, download=True)
    train_targets = np.array(train_data.targets)
    data_size = len(train_targets)
    random_index = random.sample(range(data_size), int(data_size*label_noise))
    random_part = train_targets[random_index]
    np.random.shuffle(random_part)
    train_targets[random_index] = random_part
    train_data.targets = train_targets.tolist()

    noise_data = dataset(data_root, train=False, transform=eval_transform, download=True)
    noise_data.targets = random_part.tolist()
    noise_data.data = train_data.data[random_index]

    test_data = dataset(data_root, train=False, transform=eval_transform)

elif 'svhn' in data_name:
    train_data = dataset(data_root, split='train', transform=train_transform, download=True)
    train_targets = np.array(train_data.labels)
    data_size = len(train_targets)
    random_index = random.sample(range(data_size), int(data_size * label_noise))
    random_part = train_targets[random_index]
    np.random.shuffle(random_part)
    train_targets[random_index] = random_part
    train_data.labels = train_targets.tolist()

    noise_data = dataset(data_root, split='test', transform=eval_transform, download=True)
    noise_data.labels = random_part.tolist()
    noise_data.data = train_data.data[random_index]

    test_data = dataset(data_root, split='test', transform=eval_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=eval_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=True, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# if load_model:
#     model.load_state_dict(torch.load(load_model_path))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs', data_name, "{}".format(model_name))
if bias:
    save_path = os.path.join(save_path, "bias")
else:
    save_path = os.path.join(save_path, "nobias")

if not os.path.exists(save_path):
    os.makedirs(save_path)
np.savez(os.path.join(save_path, "label_noise.npz"), index=random_index, value=random_part)
writer = SummaryWriter(log_dir=os.path.join(save_path, "log"), flush_secs=30)

wrapper.train()

for id_epoch in range(train_epoch):
    # train loop
    train_loss = 0
    train_acc = 0
    train_size = 0
    for id_batch, (inputs, targets) in enumerate(train_loader):
        loss, acc, correct = wrapper.train_on_batch(inputs, targets)
        train_loss += loss
        train_acc += correct
        train_size += len(targets)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
              format(id_epoch+1, train_epoch, id_batch+1, len(train_loader), loss, acc))
    train_loss /= id_batch
    train_acc /= train_size
    writer.add_scalar("train acc", train_acc, id_epoch+1)
    writer.add_scalar("train loss", train_loss, id_epoch+1)

    # eval
    wrapper.eval()
    test_loss, test_acc = wrapper.eval_all(test_loader)
    noise_loss, noise_acc = wrapper.eval_all(noise_loader)
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
    writer.add_scalar("test acc", test_acc, id_epoch+1)
    writer.add_scalar("test loss", test_loss, id_epoch+1)
    writer.add_scalar("noise acc", noise_acc, id_epoch+1)
    writer.add_scalar("noise loss", noise_loss, id_epoch+1)
    state = {
        'net': model.state_dict(),
        'optim': optimizer.state_dict(),
        'acc': test_acc,
        'epoch': id_epoch
    }
    torch.save(state, os.path.join(save_path, "ckpt.pkl"))

    if id_epoch % 1 == 0:
        test_energy = wrapper.predict_line_fft(test_loader, delta_h, nb_interpolation)
        avg_test_energy = np.mean(test_energy[:500]**2, axis=(0, 1))
        writer.add_scalars("test energy", {"{}".format(i): _ for i, _ in enumerate(avg_test_energy)}, id_epoch+1)

        pert_energy = wrapper.predict_line_fft(noise_loader, delta_h, nb_interpolation)
        avg_pert_energy = np.mean(pert_energy[:500]**2, axis=(0, 1))
        writer.add_scalars("pert energy", {"{}".format(i): _ for i, _ in enumerate(avg_pert_energy)}, id_epoch+1)

        train_energy = wrapper.predict_line_fft(train_loader, delta_h, nb_interpolation)
        avg_train_energy = np.mean(train_energy[:500]**2, axis=(0, 1))
        writer.add_scalars("train energy", {"{}".format(i): _ for i, _ in enumerate(avg_train_energy)}, id_epoch+1)
    print()
    # return to train state.
    wrapper.train()
writer.close()
