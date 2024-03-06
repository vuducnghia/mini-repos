import numpy as np
import math
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from time import time


class PruningModule(Module):
    # su dung do lech chuan de lam threshold
    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold: {threshold} for layer {name}')
                module.prune(threshold)


class MaskedLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter(name='bias', param=None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # thay vì nhân thẳng weight với input thì nhân với mask trước (phần tử nào của mask bị bắt giảm thì = 0)
        return F.linear(input=input, weight=self.weight * self.mask, bias=self.bias)

    def prune(self, threshhold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        # convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshhold, 0, mask)

        # apply weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
USE_CUDA = True
SEED = 42
LOG_AFTER = 10  # How many batches to wait before logging training status
LOG_FILE = 'log_prunting.txt'
SENSITIVITY = 2  # Sensitivity value that is multiplied to layer's std in order to get threshold value

# Control Seed
torch.manual_seed(SEED)

# Select Device
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# Train loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=False, **kwargs)

model = LeNet(mask=True).to(device)
# Define optimizer with Adam function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()


def train(model):
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % LOG_AFTER == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
    return model


def test(model):
    start_time = time()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(
            f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%). Total time = {time() - start_time}')

    return accuracy


def save_log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


model = train(model)
accuracy = test(model)
save_log(LOG_FILE, f"initial_accuracy {accuracy}")
torch.save(model, f"save_models/initial_model.ptmodel")


# Print number of non-zeros weight in model

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(
            f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(
        f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')


print_nonzeros(model)

# Pruning
model.prune_by_std(SENSITIVITY)
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer

model = train(model)
accuracy = test(model)
print_nonzeros(model)
save_log(LOG_FILE, f"accuracy_after_retraining {accuracy}")
torch.save(model, f"save_models/model_after_retraining.ptmodel")
