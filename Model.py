import torch.nn as nn

HIDDEN_LAYER_0_SIZE = 1100
HIDDEN_LAYER_1_SIZE = 70

IN_CHANNEL_0 = 1
OUT_CHANNEL_0 = 32
OUT_CHANNEL_1 = 64
OUT_CHANNEL_2 = 43

OUTPUT_SIZE = 30


def get_volume(in_size, kernel_size, padding, stride):
    return int((in_size - kernel_size + 2 * padding) / stride) + 1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Conv2d(IN_CHANNEL_0, OUT_CHANNEL_0, kernel_size=7, stride=1, padding=1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv1 = nn.Conv2d(OUT_CHANNEL_0, OUT_CHANNEL_1, kernel_size=7, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv2d(OUT_CHANNEL_1, OUT_CHANNEL_2, kernel_size=7, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1))
        self.dropout = nn.Dropout()

        dim1_conv0_pool0 = int((get_volume(161, 7, 1, 1) + 1) / 2)
        dim1_conv1_pool1 = int((get_volume(dim1_conv0_pool0, 7, 1, 1) + 1) / 2)
        dim1_conv2_pool2 = int(get_volume(dim1_conv1_pool1, 7, 1, 1) / 2)
        dim2_conv0_pool0 = int((get_volume(101, 7, 1, 1) + 1) / 2)
        dim2_conv1_pool1 = int((get_volume(dim2_conv0_pool0, 7, 1, 1) + 1) / 2)
        dim2_conv2_pool2 = int((get_volume(dim2_conv1_pool1, 7, 1, 1) + 1) / 2)
        self.dim1 = dim1_conv2_pool2
        self.dim2 = dim2_conv2_pool2
        self.dim3 = OUT_CHANNEL_2
        self.fc0 = nn.Linear(self.dim1 * self.dim2 * self.dim3, HIDDEN_LAYER_0_SIZE)
        self.fc1 = nn.Linear(HIDDEN_LAYER_0_SIZE, HIDDEN_LAYER_1_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_1_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = nn.functional.relu(self.conv0(x))
        x = self.pool0(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, self.dim1 * self.dim2 * self.dim3)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc0(x))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # no need for softmax because of optimizer
        return x
