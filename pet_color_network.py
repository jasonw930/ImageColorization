import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# v1: 64 256 32 2
class Net_v1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 256, 5, stride=2, padding=2),  # 128 -> 64
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 2, 1, stride=1, padding=0),  # 32 -> 32
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

    def load_data(self):
        dataset_x = np.load('pet_color_x.npy', allow_pickle=True)
        dataset_x = torch.tensor(dataset_x).float() / 255
        test_size = int(len(dataset_x) * 0.1)
        self.train_x, self.test_x = dataset_x[:-test_size], dataset_x[-test_size:]

        dataset_y = np.load('pet_color_y.npy', allow_pickle=True)
        dataset_y = (torch.tensor(dataset_y).float() - 128) / 127
        self.train_y, self.test_y = dataset_y[:-test_size], dataset_y[-test_size:]

        self.optimizer = optim.Adam(net.parameters(), lr=0.005)
        self.loss_function = nn.MSELoss()


# v2: 64 128 256 512 256 64 2
class Net_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 128 -> 64
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 32 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 64, 3, stride=1, padding=1),  # 32 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 2, 1, stride=1, padding=0),  # 32 -> 32
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

    def load_data(self):
        dataset_x = np.load('pet_color_x.npy', allow_pickle=True)
        dataset_x = torch.tensor(dataset_x).float() / 255
        test_size = int(len(dataset_x) * 0.1)
        self.train_x, self.test_x = dataset_x[:-test_size], dataset_x[-test_size:]

        dataset_y = np.load('pet_color_y.npy', allow_pickle=True)
        dataset_y = (torch.tensor(dataset_y).float() - 128) / 127
        self.train_y, self.test_y = dataset_y[:-test_size], dataset_y[-test_size:]

        self.optimizer = optim.Adam(net.parameters(), lr=0.005)
        self.loss_function = nn.MSELoss()


# v3: 64x2 128x2 256x2 512x2 256x2 64x2 2
class Net_v3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 128 -> 64
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # 64 -> 32
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 64, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 2, 1, stride=1, padding=0),  # 32 -> 32
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

    def load_data(self):
        dataset_x = np.load('pet_color_x.npy', allow_pickle=True)
        dataset_x = torch.tensor(dataset_x).float() / 255
        test_size = int(len(dataset_x) * 0.1)
        self.train_x, self.test_x = dataset_x[:-test_size], dataset_x[-test_size:]

        dataset_y = np.load('pet_color_y.npy', allow_pickle=True)
        dataset_y = (torch.tensor(dataset_y).float() - 128) / 127
        self.train_y, self.test_y = dataset_y[:-test_size], dataset_y[-test_size:]

        self.optimizer = optim.Adam(net.parameters(), lr=0.005)
        self.loss_function = nn.MSELoss()


# v4: 64x2 128x2 256x2 512x2 256x2 128x2 64
class Net_v4(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 128 -> 64
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # 64 -> 32
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 1, stride=1, padding=0),  # 32 -> 32
            nn.Softmax2d(),
        )

    def forward(self, x):
        return self.layers(x)

    def load_data(self):
        dataset_x = np.load('pet_color_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))
        test_size = int(len(dataset_x) * 0.1)
        self.train_x, self.test_x = dataset_x[:-test_size], dataset_x[-test_size:]

        dataset_y = np.load('pet_color_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.uint8)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)
        self.train_y, self.test_y = dataset_y[:-test_size], dataset_y[-test_size:]

        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        for epoch in range(EPOCHS):
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(0, int(len(self.train_x) * train_size), BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y = self.one_hot(self.train_y[i:i+BATCH_SIZE]).to(device)

                self.zero_grad()
                result = self(batch_x)
                loss = self.loss_function(result, batch_y)
                loss.backward()
                self.optimizer.step()

                loss_total += loss
                loss_count += 1
            print('Average Train Loss: %.3f' % (loss_total / loss_count).item())

    def test(self):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(len(self.test_x))):
                result = self(self.test_x[i:i+1].to(device))
                loss = self.loss_function(result, self.one_hot(self.test_y[i:i+1]).to(device))

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


# v5: 64x2 128x2 256x2 512x2 256x2 128x2 -> 32a -> CrossEntropy
#                                        -> 32b -> CrossEntropy
class Net_v5(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 128 -> 64
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # 64 -> 32
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_a = nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(128, 32, 1, stride=1, padding=0)

    def forward(self, x):
        seq_output = self.seq_layers(x)
        return self.conv_a(seq_output), self.conv_b(seq_output)

    def load_data(self):
        dataset_x = np.load('pet_color_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))
        test_size = int(len(dataset_x) * 0.1)
        self.train_x, self.test_x = dataset_x[:-test_size], dataset_x[-test_size:]

        dataset_y = np.load('pet_color_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.int64)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)
        self.train_y, self.test_y = dataset_y[:-test_size], dataset_y[-test_size:]

        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.loss_function_a = nn.CrossEntropyLoss()
        self.loss_function_b = nn.CrossEntropyLoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        for epoch in range(EPOCHS):
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(0, int(len(self.train_x) * train_size), BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y_a = self.train_y[i:i+BATCH_SIZE, 0].to(device)
                batch_y_b = self.train_y[i:i+BATCH_SIZE, 1].to(device)

                self.zero_grad()
                result_a, result_b = self(batch_x)
                loss_a = self.loss_function_a(result_a, batch_y_a)
                loss_b = self.loss_function_b(result_b, batch_y_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()

                loss_total += loss
                loss_count += 1
            print('Average Train Loss: %.3f' % (loss_total / loss_count).item())

    def test(self, test_size=1):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(int(len(self.test_x) * test_size))):
                result_a, result_b = self(self.test_x[i:i+1].to(device))
                loss_a = self.loss_function_a(result_a, self.test_y[i:i+1, 0].to(device))
                loss_b = self.loss_function_b(result_b, self.test_y[i:i+1, 1].to(device))
                loss = loss_a + loss_b

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


# v6: 64x2 128x2 256x2 512x2 512x2 512x2 256x2 128x2 -> 32a -> CrossEntropy
#                                                    -> 32b -> CrossEntropy
class Net_v6(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 128 -> 64
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 5, stride=1, padding=2),  # 64 -> 64
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 64 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_a = nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(128, 32, 1, stride=1, padding=0)

    def forward(self, x):
        seq_output = self.seq_layers(x)
        return self.conv_a(seq_output), self.conv_b(seq_output)

    def load_data(self, drive, data_name):
        path = '/content/drive/MyDrive/AI/' if drive else ''
        dataset_x = np.load(path + data_name + '_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))

        dataset_y = np.load(path + data_name + '_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.int64)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)

        test_size = int(len(dataset_x) * 0.1)
        val_size = int((len(dataset_x) - test_size) * 0.1)

        self.train_x = dataset_x[:-test_size-val_size]
        self.val_x = dataset_x[-test_size-val_size:-test_size]
        self.test_x = dataset_x[-test_size:]

        self.train_y = dataset_y[:-test_size-val_size]
        self.val_y = dataset_y[-test_size-val_size:-test_size]
        self.test_y = dataset_y[-test_size:]

        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.loss_function_a = nn.CrossEntropyLoss()
        self.loss_function_b = nn.CrossEntropyLoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        train_len = int(len(self.train_x) * train_size)
        train_loss = []
        val_loss = []

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, train_len, BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y_a = self.train_y[i:i+BATCH_SIZE, 0].to(device)
                batch_y_b = self.train_y[i:i+BATCH_SIZE, 1].to(device)

                self.zero_grad()
                result_a, result_b = self(batch_x)
                loss_a = self.loss_function_a(result_a, batch_y_a)
                loss_b = self.loss_function_b(result_b, batch_y_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                val_x = self.val_x.to(device)
                val_y_a = self.val_y[:, 0].to(device)
                val_y_b = self.val_y[:, 1].to(device)

                with torch.no_grad():
                    result_a, result_b = self(val_x)
                    loss_a = self.loss_function_a(result_a, val_y_a)
                    loss_b = self.loss_function_b(result_b, val_y_b)
                    loss = loss_a + loss_b
                    val_loss.append(loss.item())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss)
        ax.plot(val_loss)
        ax.legend(['Train Loss', 'Validation Loss'])
        fig.show()

    def test(self, test_size=1):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(int(len(self.test_x) * test_size))):
                result_a, result_b = self(self.test_x[i:i+1].to(device))
                loss_a = self.loss_function_a(result_a, self.test_y[i:i+1, 0].to(device))
                loss_b = self.loss_function_b(result_b, self.test_y[i:i+1, 1].to(device))
                loss = loss_a + loss_b

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


# v7: 64x2 128x2 256x2 512x2 512x2 512x2 256x2 128x2 -> 32a -> CrossEntropy
#                                                    -> 32b -> CrossEntropy
# Dropout
class Net_v7(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # 128 -> 128
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 128 -> 64
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(128, 256, 5, stride=1, padding=2),  # 64 -> 64
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 64 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),  # 32 -> 32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.conv_a = nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(128, 32, 1, stride=1, padding=0)

    def forward(self, x):
        seq_output = self.seq_layers(x)
        return self.conv_a(seq_output), self.conv_b(seq_output)

    def load_data(self, drive, data_name):
        path = '/content/drive/MyDrive/AI/' if drive else ''
        dataset_x = np.load(path + data_name + '_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))

        dataset_y = np.load(path + data_name + '_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.int64)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)

        test_size = int(len(dataset_x) * 0.1)
        val_size = int((len(dataset_x) - test_size) * 0.1)

        self.train_x = dataset_x[:-test_size-val_size]
        self.val_x = dataset_x[-test_size-val_size:-test_size]
        self.test_x = dataset_x[-test_size:]

        self.train_y = dataset_y[:-test_size-val_size]
        self.val_y = dataset_y[-test_size-val_size:-test_size]
        self.test_y = dataset_y[-test_size:]

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_function_a = nn.CrossEntropyLoss()
        self.loss_function_b = nn.CrossEntropyLoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        train_len = int(len(self.train_x) * train_size)
        train_loss = []
        val_loss = []

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, train_len, BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y_a = self.train_y[i:i+BATCH_SIZE, 0].to(device)
                batch_y_b = self.train_y[i:i+BATCH_SIZE, 1].to(device)

                self.zero_grad()
                result_a, result_b = self(batch_x)
                loss_a = self.loss_function_a(result_a, batch_y_a)
                loss_b = self.loss_function_b(result_b, batch_y_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                val_x = self.val_x.to(device)
                val_y_a = self.val_y[:, 0].to(device)
                val_y_b = self.val_y[:, 1].to(device)

                with torch.no_grad():
                    result_a, result_b = self(val_x)
                    loss_a = self.loss_function_a(result_a, val_y_a)
                    loss_b = self.loss_function_b(result_b, val_y_b)
                    loss = loss_a + loss_b
                    val_loss.append(loss.item())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss)
        ax.plot(val_loss)
        ax.legend(['Train Loss', 'Validation Loss'])
        fig.show()

    def test(self, test_size=1):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(int(len(self.test_x) * test_size))):
                result_a, result_b = self(self.test_x[i:i+1].to(device))
                loss_a = self.loss_function_a(result_a, self.test_y[i:i+1, 0].to(device))
                loss_b = self.loss_function_b(result_b, self.test_y[i:i+1, 1].to(device))
                loss = loss_a + loss_b

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


# v8: 64x2 128x2 256x2 512x2 256 64 -> 32a -> CrossEntropy
#                                   -> 32b -> CrossEntropy
# Dropout, ConvTranspose2d
class Net_v8(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 128 -> 64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # 64 -> 32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 32 -> 16
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 16 -> 16
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),  # 16 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.conv_a = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(64, 32, 1, stride=1, padding=0)

    def forward(self, x):
        seq_output = self.seq_layers(x)
        return self.conv_a(seq_output), self.conv_b(seq_output)

    def load_data(self, drive, data_name):
        path = '/content/drive/MyDrive/AI/' if drive else ''
        dataset_x = np.load(path + data_name + '_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))

        dataset_y = np.load(path + data_name + '_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.int64)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)

        test_size = int(len(dataset_x) * 0.1)
        val_size = int((len(dataset_x) - test_size) * 0.1)

        self.train_x = dataset_x[:-test_size-val_size]
        self.val_x = dataset_x[-test_size-val_size:-test_size]
        self.test_x = dataset_x[-test_size:]

        self.train_y = dataset_y[:-test_size-val_size]
        self.val_y = dataset_y[-test_size-val_size:-test_size]
        self.test_y = dataset_y[-test_size:]

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_function_a = nn.CrossEntropyLoss()
        self.loss_function_b = nn.CrossEntropyLoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        train_len = int(len(self.train_x) * train_size)
        train_loss = []
        val_loss = []

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, train_len, BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y_a = self.train_y[i:i+BATCH_SIZE, 0].to(device)
                batch_y_b = self.train_y[i:i+BATCH_SIZE, 1].to(device)

                self.zero_grad()
                result_a, result_b = self(batch_x)
                loss_a = self.loss_function_a(result_a, batch_y_a)
                loss_b = self.loss_function_b(result_b, batch_y_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                val_x = self.val_x.to(device)
                val_y_a = self.val_y[:, 0].to(device)
                val_y_b = self.val_y[:, 1].to(device)

                with torch.no_grad():
                    result_a, result_b = self(val_x)
                    loss_a = self.loss_function_a(result_a, val_y_a)
                    loss_b = self.loss_function_b(result_b, val_y_b)
                    loss = loss_a + loss_b
                    val_loss.append(loss.item())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss)
        ax.plot(val_loss)
        ax.legend(['Train Loss', 'Validation Loss'])
        fig.show()

    def test(self, test_size=1):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(int(len(self.test_x) * test_size))):
                result_a, result_b = self(self.test_x[i:i+1].to(device))
                loss_a = self.loss_function_a(result_a, self.test_y[i:i+1, 0].to(device))
                loss_b = self.loss_function_b(result_b, self.test_y[i:i+1, 1].to(device))
                loss = loss_a + loss_b

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


# v9: L2RGB vgg16_bn(64x2 P 128x2 P 256x3 P 512x3 P 512x3) 64 64T -> 32Ta -> CrossEntropy
#                                                                 -> 32Tb -> CrossEntropy
# L2 Regularization, val_loss ≈ 3.67
class Net_v9(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg_layers = models.vgg16_bn(pretrained=True, progress=True).features[:34+9]
        self.vgg_layers = self.vgg_layers.requires_grad_(False)

        self.convt = []
        in_channels = 512
        for layers in ['64', '64T']:
            transpose = layers[-1] == 'T'
            out_channels = int(layers[:-1] if transpose else layers)

            if transpose:
                self.convt += [
                    nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(0.5)
                ]
            else:
                self.convt += [
                    nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(0.5)
                ]
            in_channels = out_channels
        self.convt = nn.Sequential(*self.convt)

        self.convt_a = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.convt_b = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.vgg_layers(x)
        x = self.convt(x)

        a = self.convt_a(x)
        b = self.convt_b(x)
        return a, b

    def load_data(self, drive, data_name, t_size=0.1, v_size=0.1):
        path = '/content/drive/MyDrive/AI/' if drive else ''
        dataset_x = np.load(path + data_name + '_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))

        dataset_y = np.load(path + data_name + '_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.int64)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)

        test_size = int(len(dataset_x) * t_size)
        val_size = int((len(dataset_x) - test_size) * v_size)

        self.train_x = dataset_x[:-test_size-val_size]
        self.val_x = dataset_x[-test_size-val_size:-test_size]
        self.test_x = dataset_x[-test_size:]

        self.train_y = dataset_y[:-test_size-val_size]
        self.val_y = dataset_y[-test_size-val_size:-test_size]
        self.test_y = dataset_y[-test_size:]

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_function_a = nn.CrossEntropyLoss()
        self.loss_function_b = nn.CrossEntropyLoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        train_len = int(len(self.train_x) * train_size)
        train_loss = []
        val_loss = []

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, train_len, BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y_a = self.train_y[i:i+BATCH_SIZE, 0].to(device)
                batch_y_b = self.train_y[i:i+BATCH_SIZE, 1].to(device)

                self.zero_grad()
                result_a, result_b = self(batch_x)
                loss_a = self.loss_function_a(result_a, batch_y_a)
                loss_b = self.loss_function_b(result_b, batch_y_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                val_x = self.val_x.to(device)
                val_y_a = self.val_y[:, 0].to(device)
                val_y_b = self.val_y[:, 1].to(device)

                with torch.no_grad():
                    result_a, result_b = self(val_x)
                    loss_a = self.loss_function_a(result_a, val_y_a)
                    loss_b = self.loss_function_b(result_b, val_y_b)
                    loss = loss_a + loss_b
                    val_loss.append(loss.item())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss)
        ax.plot(val_loss)
        ax.legend(['Train Loss', 'Validation Loss'])
        fig.show()
        print(train_loss)
        print(val_loss)

    def test(self, test_size=1):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(int(len(self.test_x) * test_size))):
                result_a, result_b = self(self.test_x[i:i+1].to(device))
                loss_a = self.loss_function_a(result_a, self.test_y[i:i+1, 0].to(device))
                loss_b = self.loss_function_b(result_b, self.test_y[i:i+1, 1].to(device))
                loss = loss_a + loss_b

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


# v10: L2RGB vgg19_bn(64x2 P 128x2 P 256x4 P 512x4 P 512x4) 64 64T -> 32Ta -> CrossEntropy
#                                                                  -> 32Tb -> CrossEntropy
# L2 Regularization, val_loss ≈ 3.67
class Net_v10(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg_layers = models.vgg19_bn(pretrained=True, progress=True).features[:6+1+6+1+12+1+12+1+12]
        self.vgg_layers = self.vgg_layers.requires_grad_(False)

        self.convt = []
        in_channels = 512
        for layers in ['64', '64T']:
            transpose = layers[-1] == 'T'
            out_channels = int(layers[:-1] if transpose else layers)

            if transpose:
                self.convt += [
                    nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(0.5)
                ]
            else:
                self.convt += [
                    nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(0.5)
                ]
            in_channels = out_channels
        self.convt = nn.Sequential(*self.convt)

        self.convt_a = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.convt_b = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.vgg_layers(x)
        x = self.convt(x)

        a = self.convt_a(x)
        b = self.convt_b(x)
        return a, b

    def load_data(self, drive, data_name, t_size=0.1, v_size=0.1):
        path = '/content/drive/MyDrive/AI/' if drive else ''
        dataset_x = np.load(path + data_name + '_x.npy', allow_pickle=True)
        dataset_x = torch.tensor((dataset_x / 255).astype(np.float32))

        dataset_y = np.load(path + data_name + '_y.npy', allow_pickle=True)
        dataset_y = (dataset_y / 8).astype(np.int64)
        dataset_y = torch.tensor(dataset_y)

        print(dataset_x.dtype)
        print(dataset_x.shape)
        print(dataset_y.dtype)
        print(dataset_y.shape)

        test_size = int(len(dataset_x) * t_size)
        val_size = int((len(dataset_x) - test_size) * v_size)

        self.train_x = dataset_x[:-test_size-val_size]
        self.val_x = dataset_x[-test_size-val_size:-test_size]
        self.test_x = dataset_x[-test_size:]

        self.train_y = dataset_y[:-test_size-val_size]
        self.val_y = dataset_y[-test_size-val_size:-test_size]
        self.test_y = dataset_y[-test_size:]

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_function_a = nn.CrossEntropyLoss()
        self.loss_function_b = nn.CrossEntropyLoss()

    def one_hot(self, dataset):
        dataset = np.array(dataset)
        dataset_onehot = np.zeros((dataset.shape[0], 64, 32, 32)).astype(np.uint8)

        x_coords = np.repeat(np.arange(32), 32)
        y_coords = np.tile(np.arange(32), 32)
        for i in range(dataset_onehot.shape[0]):
            dataset_onehot[[i]*32*32, dataset[i, 0].reshape(-1), x_coords, y_coords] = 1
            dataset_onehot[[i]*32*32, dataset[i, 1].reshape(-1)+32, x_coords, y_coords] = 1

        return torch.tensor(dataset_onehot)

    def train(self, BATCH_SIZE, EPOCHS, train_size=1):
        train_len = int(len(self.train_x) * train_size)
        train_loss = []
        val_loss = []

        for epoch in range(EPOCHS):
            for i in tqdm(range(0, train_len, BATCH_SIZE)):
                batch_x = self.train_x[i:i+BATCH_SIZE].to(device)
                batch_y_a = self.train_y[i:i+BATCH_SIZE, 0].to(device)
                batch_y_b = self.train_y[i:i+BATCH_SIZE, 1].to(device)

                self.zero_grad()
                result_a, result_b = self(batch_x)
                loss_a = self.loss_function_a(result_a, batch_y_a)
                loss_b = self.loss_function_b(result_b, batch_y_b)
                loss = loss_a + loss_b
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                val_x = self.val_x.to(device)
                val_y_a = self.val_y[:, 0].to(device)
                val_y_b = self.val_y[:, 1].to(device)

                with torch.no_grad():
                    result_a, result_b = self(val_x)
                    loss_a = self.loss_function_a(result_a, val_y_a)
                    loss_b = self.loss_function_b(result_b, val_y_b)
                    loss = loss_a + loss_b
                    val_loss.append(loss.item())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss)
        ax.plot(val_loss)
        ax.legend(['Train Loss', 'Validation Loss'])
        fig.show()
        print(train_loss)
        print(val_loss)

    def test(self, test_size=1):
        with torch.no_grad():
            loss_total = 0
            loss_count = 0
            for i in tqdm(range(int(len(self.test_x) * test_size))):
                result_a, result_b = self(self.test_x[i:i+1].to(device))
                loss_a = self.loss_function_a(result_a, self.test_y[i:i+1, 0].to(device))
                loss_b = self.loss_function_b(result_b, self.test_y[i:i+1, 1].to(device))
                loss = loss_a + loss_b

                loss_total += loss
                loss_count += 1
        print('Average Test Loss: %.3f' % (loss_total / loss_count).item())


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

net = Net_v8().to(device)
net.load_data(False, 'land')

# One epoch at lr=0.01 is pretty good
# Five epochs at lr=0.005 is pretty good


def train(BATCH_SIZE, EPOCHS):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(net.train_x), BATCH_SIZE)):
            batch_x = net.train_x[i:i+BATCH_SIZE].to(device)
            batch_y = net.train_y[i:i+BATCH_SIZE].to(device)

            net.zero_grad()
            result = net(batch_x)
            loss = net.loss_function(result, batch_y)
            loss.backward()
            net.optimizer.step()


def test():
    with torch.no_grad():
        loss_total = 0
        loss_count = 0
        for i in tqdm(range(len(net.test_x))):
            result = net(net.test_x[i:i+1].to(device))
            loss = net.loss_function(result, net.test_y[i:i+1].to(device))

            loss_total += loss
            loss_count += 1
    print('Average Loss:', loss_total / loss_count)


net.train(1, 3, train_size=1)
