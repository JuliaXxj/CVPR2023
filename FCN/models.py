import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import logging
from datetime import datetime
import copy

# to get activation
ACTIVATION = None


def get_activation(name, tensor_logger, detach, is_lastlayer=False):
    if is_lastlayer:
        def hook(model, input, output):
            raw = torch.flatten(output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            # use argmax instead of broadcasting just in case comparing floating point is finicky

            mask = np.zeros(raw.shape, dtype=bool)

            mask[np.arange(raw.shape[0]), raw.argmax(axis=1)] = 1

            tensor_logger[name] = np.concatenate((tensor_logger[name], mask),
                                                 axis=0) if name in tensor_logger else mask

        return hook

    if detach:
        def hook(model, input, output):
            raw = torch.flatten(
                output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            raw = raw > 0
            logging.debug("{}, {}".format(name, raw.shape))
            tensor_logger[name] = np.concatenate((tensor_logger[name], raw),
                                                 axis=0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)

        return hook
    else:
        # keep the gradient, so cannot convert to bit here
        def hook(model, input, output):
            raw = torch.sigmoid(torch.flatten(
                output, start_dim=1, end_dim=-1))
            logging.debug("{}, {}".format(name, raw.shape))
            tensor_logger[name] = torch.cat((tensor_logger[name], raw),
                                            axis=0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)

        return hook


def get_gradient(name, gradient_logger, detach):
    def hook(model, grad_input, grad_output):
        raw = grad_output
        assert (len(raw) == 1)
        raw = raw[0].cpu().detach().numpy()
        gradient_logger[name] = np.concatenate((gradient_logger[name], raw), axis=0) if name in gradient_logger else raw

    return hook


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.tensor_log = {}
        self.gradient_log = {}
        self.hooks = []
        self.bw_hooks = []

    def reset_hooks(self):
        self.tensor_log = {}
        for h in self.hooks:
            h.remove()

    def reset_bw_hooks(self):
        self.input_labels = None
        self.gradient_log = {}
        for h in self.bw_hooks:
            h.remove()

    def register_log(self, detach):
        raise NotImplementedError

    def register_gradient(self, detach):
        raise NotImplementedError

    def model_savename(self):
        raise NotImplementedError

    def get_pattern(self, input, layers, device, flatten=True):
        self.eval()
        self.register_log()
        self.forward(input.to(device))
        tensor_log = copy.deepcopy(self.tensor_log)
        if flatten:
            return np.concatenate([tensor_log[l] for l in layers], axis=1)
        return tensor_log


class TinyCNN(BaseNet):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1152, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #         x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #         x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.conv1.register_forward_hook(get_activation('conv1', self.tensor_log, detach)))
        self.hooks.append(self.conv2.register_forward_hook(get_activation('conv2', self.tensor_log, detach)))
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.conv1.register_backward_hook(get_gradient('conv1', self.gradient_log, detach)))
        self.bw_hooks.append(self.conv2.register_backward_hook(get_gradient('conv2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))

    def model_savename(self, tag=""):
        return "TinyCNN" + tag + datetime.now().strftime("%H-%M-%S")


class FeedforwardNeuralNetModel(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, 256, bias=bias)
        self.fc2 = nn.Linear(256, 128, bias=bias)
        self.fc3 = nn.Linear(128, 64, bias=bias)
        self.fc4 = nn.Linear(64, output_dim, bias=bias)

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))
        self.hooks.append(self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach)))
        # self.hooks.append(self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach)))
        self.hooks.append(
            self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach, is_lastlayer=True)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc4.register_backward_hook(get_gradient('fc4', self.gradient_log, detach)))

    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28 * 28)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        #    out = F.log_softmax(out, dim=1)
        return out

    def model_savename(self):
        return "FFN" + datetime.now().strftime("%H-%M-%S")


class DummyFCN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, 10, bias=bias)
        self.fc2 = nn.Linear(10, 10, bias=bias)
        self.fc3 = nn.Linear(10, output_dim, bias=bias)

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))

        self.hooks.append(
            self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach, is_lastlayer=True)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))


    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28 * 28)))
        out = self.fc2(out)
        out = self.fc3(out)
        #    out = F.log_softmax(out, dim=1)
        return out

    def model_savename(self):
        return "DummyFCN" + datetime.now().strftime("%H-%M-%S")

class VNN_FFN_RELU_2(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VNN_FFN_RELU_2, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))
        self.hooks.append(
            self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach, is_lastlayer=True)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))


    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28 * 28)))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        #    out = F.log_softmax(out, dim=1)
        return out
    
    def update_all_weights(self, names_in_order, new_weights):
        if len(names_in_order) != 3 or len(new_weights) != 3:
            raise Exception("Expected the number of layer names or new weights to be 3.")
        self.fc1.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[0]]).float())
        self.fc2.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[1]]).float())
        self.fc3.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[2]]).float())
        
    def update_all_bias(self, names_in_order, new_bias):
        if len(names_in_order) != 3 or len(new_bias) != 3:
            raise Exception("Expected the number of layer names or new bias to be 3.")
        self.fc1.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[0]]).float())
        self.fc2.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[1]]).float())
        self.fc3.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[2]]).float())

class VNN_FFN_RELU_4(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VNN_FFN_RELU_4, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))
        self.hooks.append(self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach)))
        self.hooks.append(self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach)))
        self.hooks.append(
            self.fc5.register_forward_hook(get_activation('fc5', self.tensor_log, detach, is_lastlayer=True)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc4.register_backward_hook(get_gradient('fc4', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc5.register_backward_hook(get_gradient('fc5', self.gradient_log, detach)))


    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28 * 28)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        #    out = F.log_softmax(out, dim=1)
        return out
    
    def update_all_weights(self, names_in_order, new_weights):
        if len(names_in_order) != 5 or len(new_weights) != 5:
            raise Exception("Expected the number of layer names or new weights to be 5.")
        self.fc1.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[0]]).float())
        self.fc2.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[1]]).float())
        self.fc3.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[2]]).float())
        self.fc4.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[3]]).float())
        self.fc5.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[4]]).float())
        
    def update_all_bias(self, names_in_order, new_bias):
        if len(names_in_order) != 5 or len(new_bias) != 5:
            raise Exception("Expected the number of layer names or new bias to be 5.")
        self.fc1.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[0]]).float())
        self.fc2.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[1]]).float())
        self.fc3.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[2]]).float())
        self.fc4.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[3]]).float())
        self.fc5.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[4]]).float())
 

class VNN_FFN_RELU_6(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VNN_FFN_RELU_6, self).__init__()

        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))
        self.hooks.append(self.fc3.register_forward_hook(get_activation('fc3', self.tensor_log, detach)))
        self.hooks.append(self.fc4.register_forward_hook(get_activation('fc4', self.tensor_log, detach)))
        self.hooks.append(self.fc5.register_forward_hook(get_activation('fc5', self.tensor_log, detach)))
        self.hooks.append(self.fc6.register_forward_hook(get_activation('fc6', self.tensor_log, detach)))
        self.hooks.append(
            self.fc7.register_forward_hook(get_activation('fc7', self.tensor_log, detach, is_lastlayer=True)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc3.register_backward_hook(get_gradient('fc3', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc4.register_backward_hook(get_gradient('fc4', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc5.register_backward_hook(get_gradient('fc5', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc6.register_backward_hook(get_gradient('fc6', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc7.register_backward_hook(get_gradient('fc7', self.gradient_log, detach)))


    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, 28 * 28)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))                  
        out = self.fc7(out)
        #    out = F.log_softmax(out, dim=1)
        return out
    
    def update_all_weights(self, names_in_order, new_weights):
        if len(names_in_order) != 7 or len(new_weights) != 7:
            raise Exception("Expected the number of layer names or new weights to be 7.")
        self.fc1.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[0]]).float())
        self.fc2.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[1]]).float())
        self.fc3.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[2]]).float())
        self.fc4.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[3]]).float())
        self.fc5.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[4]]).float())
        self.fc6.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[5]]).float())
        self.fc7.weight = torch.nn.Parameter(torch.from_numpy( new_weights[names_in_order[6]]).float())
        
    def update_all_bias(self, names_in_order, new_bias):
        if len(names_in_order) != 7 or len(new_bias) != 7:
            raise Exception("Expected the number of layer names or new bias to be 7.")
        self.fc1.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[0]]).float())
        self.fc2.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[1]]).float())
        self.fc3.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[2]]).float())
        self.fc4.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[3]]).float())
        self.fc5.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[4]]).float())
        self.fc6.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[5]]).float())
        self.fc7.bias = torch.nn.Parameter(torch.from_numpy( new_bias[names_in_order[6]]).float())
    

class PatternClassifier(BaseNet):
    def __init__(self, input_dim, max_unit, output_dim):
        super(PatternClassifier, self).__init__()
        self.max_unit = max_unit
        # Linear function
        self.fc1 = nn.Linear(self.max_unit, output_dim, bias=False)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)

    def register_log(self):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log)))
        # self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log)))

    def forward(self, x):
        # out = F.relu(self.fc1(x))
        out = self.fc1(x[:, :self.max_unit])
        # out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

    def model_savename(self):
        return "PatternClasffier" + datetime.now().strftime("%H:%M:%S")


def fgsm_attack(image, eps, data_grad):
    sign_data_grad = data_grad.sign()
    pertubed_image = image + eps * sign_data_grad
    pertubed_image = torch.clamp(pertubed_image, 0, 1)
    return pertubed_image




# https://github.com/pytorch/examples/tree/main/mnist

class NewTinyCNN(BaseNet):
    def __init__(self):
        super(NewTinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #         x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #         x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.conv1.register_forward_hook(get_activation('conv1', self.tensor_log, detach)))
        self.hooks.append(self.conv2.register_forward_hook(get_activation('conv2', self.tensor_log, detach)))
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.conv1.register_backward_hook(get_gradient('conv1', self.gradient_log, detach)))
        self.bw_hooks.append(self.conv2.register_backward_hook(get_gradient('conv2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))

    def model_savename(self, tag=""):
        return "NewTinyCNN" + tag + datetime.now().strftime("%H:%M:%S")

class MNISTNet(BaseNet):
    def __init__(self, bias=True):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=bias)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128, bias=bias)
        self.fc2 = nn.Linear(128, 10, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
        return x

    def register_log(self, detach=True):
        self.reset_hooks()
        # first layer should not make any difference?
        self.hooks.append(self.conv1.register_forward_hook(get_activation('conv1', self.tensor_log, detach)))
        self.hooks.append(self.conv2.register_forward_hook(get_activation('conv2', self.tensor_log, detach)))
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1', self.tensor_log, detach)))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2', self.tensor_log, detach)))

    def register_gradient(self, detach=True):
        self.reset_bw_hooks()
        # first layer should not make any difference?
        self.bw_hooks.append(self.conv1.register_backward_hook(get_gradient('conv1', self.gradient_log, detach)))
        self.bw_hooks.append(self.conv2.register_backward_hook(get_gradient('conv2', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc1.register_backward_hook(get_gradient('fc1', self.gradient_log, detach)))
        self.bw_hooks.append(self.fc2.register_backward_hook(get_gradient('fc2', self.gradient_log, detach)))


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # output = F.softmax(x, dim=1)
        return x
