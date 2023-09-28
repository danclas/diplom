# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Imports

from dataset import load_dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from metrics import statistics, preproc

from ray import air, tune
from ray.air import session

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# +
# import os
# os.environ['CUDA_LAUNCH_BLOCKING']="0"
# -

# # Transformer + Position Encoding

# +
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):   
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, d_model, h, d_ff, num_layers, n_channels, n_classes, dropout):
        """
        :param d_model: количество ожидаемых features на входе трансформера
        :param h: количество heads в multiheadattention
        :param d_ff: количесвто нейроннов в полносвязной сети трансформера
        :param num_layers: количество слоёв в трансформере
        :param n_channels: количество каналов
        :param n_classes: количество классов
        :param dropout:
        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_channels = n_channels
        self.pe = PositionalEncoding(d_model, dropout=0.1)
        
        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.h, 
            dim_feedforward=self.d_ff, 
            dropout=self.dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)
        
        self.encoder = nn.Linear(n_channels, d_model)
        self.decoder = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        out = self.encoder(x) * math.sqrt(self.d_model)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = self.decoder(out)
        out = out.permute((1, 2, 0))
        return out


# -

# # Parts of the UNet model

# +
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=8, stride=2, padding=3)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



# -

# # Standart implementation of UNet

# +
""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, 8)
        self.up4 = Up(8, 4)
        self.outc = OutConv(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -

# # Second version of UNet with scale factor

# +
""" Full assembly of the parts to form the complete network """

class UNetV2(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers, start_scale):
        """
        :params start_scale: число, с которого начнётся уменьшение,
        Если start_sclae = 2, то Down(2, 4), Down(4, 8), Down(8, 16) и тд
        """
        super(UNetV2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers

        self.inc = DoubleConv(n_channels, start_scale)

        if (n_layers > 1):
            self.encoder = nn.ModuleList(
                [Down(start_scale * (2 ** (i - 1)), start_scale * (2 ** i)) for i in range(1, n_layers)])
            self.decoder = nn.ModuleList(
                [Up(start_scale * (2 ** i), start_scale * (2 ** (i - 1))) for i in reversed(range(1, n_layers))])

        self.outc = OutConv(start_scale, n_classes)

    def forward(self, x):
        x_list = [self.inc(x)]
        if (self.n_layers > 1):
            for down in self.encoder:
                x_list.append(down(x_list[-1]))
        if (self.n_layers > 1):
            for up in self.decoder:
                x_list.append(up(x_list.pop(), x_list.pop()))

        logits = self.outc(x_list[0])
        return logits


# -


# # Version of UNet with Transformer in beginning.

class UNetInT(nn.Module):
    def __init__(self, n_channels, n_classes, unet_n_layers, start_scale, d_model, h, d_ff, num_layers, dropout):
        """
        :param n_channels: количество входных каналов
        :param n_classes: количество выходных каналов (классов)
        :param start_scale: с какого количества каналов начинает работать Unet
        :param d_model: количество ожидаемых features на входе трансформера
        :param h: количество heads в multiheadattention
        :param d_ff: количесвто нейроннов в полносвязной сети трансформера
        :param num_layers: количество слоёв в трансформере
        :param dropout:
        """
        super(UNetInT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.unet_n_layers = unet_n_layers
        
        self.transformer = Transformer(d_model, h, d_ff, num_layers, n_channels, n_channels, dropout=dropout)
        
        self.inc = DoubleConv(n_channels, start_scale)
        
        if(unet_n_layers > 1):
            self.encoder = nn.ModuleList([Down(start_scale * (2 ** (i - 1)), start_scale * (2 ** i)) for i in range(1, unet_n_layers)])
            self.decoder = nn.ModuleList([Up(start_scale * (2 ** i), start_scale * (2 ** (i - 1))) for i in reversed(range(1, unet_n_layers))])

        self.outc = OutConv(start_scale, n_classes)

    def forward(self, x):
        x = self.transformer(x.permute(0, 2, 1))
        x_list = [self.inc(x)]
        if self.unet_n_layers > 1:
            for down in self.encoder:
                x_list.append(down(x_list[-1]))
        
        if self.unet_n_layers > 1:
            for up in self.decoder:
                x_list.append(up(x_list.pop(), x_list.pop()))

        logits = self.outc(x_list[0])
        return logits


# # Version of UNet with Transformer in middle.

class UNetMiddleT(nn.Module):
    def __init__(self, n_channels, n_classes, unet_n_layers, start_scale, d_model, h, d_ff, num_layers, dropout):
        """
        :param n_channels: количество входных каналов
        :param n_classes: количество выходных каналов (классов)
        :param start_scale: с какого количества каналов начинает работать Unet
        :param d_model: количество ожидаемых features на входе трансформера
        :param h: количество heads в multiheadattention
        :param d_ff: количесвто нейроннов в полносвязной сети трансформера
        :param num_layers: количество слоёв в трансформере
        :param dropout:
        """
        super(UNetMiddleT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.unet_n_layers = unet_n_layers

        self.transformer = Transformer(d_model, h, d_ff, num_layers, start_scale * (2 ** (unet_n_layers - 1)),
                                       start_scale * (2 ** (unet_n_layers - 1)), dropout=dropout)

        self.inc = DoubleConv(n_channels, start_scale)

        if (unet_n_layers > 1):
            self.encoder = nn.ModuleList(
                [Down(start_scale * (2 ** (i - 1)), start_scale * (2 ** i)) for i in range(1, unet_n_layers)])
            self.decoder = nn.ModuleList(
                [Up(start_scale * (2 ** i), start_scale * (2 ** (i - 1))) for i in reversed(range(1, unet_n_layers))])

        self.outc = OutConv(start_scale, n_classes)

    def forward(self, x):
        x_list = [self.inc(x)]
        
        if self.unet_n_layers > 1:
            for down in self.encoder:
                x_list.append(down(x_list[-1]))
                
        x_list.append(self.transformer(x_list.pop().permute(0, 2, 1)))
        
        if self.unet_n_layers > 1:
            for up in self.decoder:
                x_list.append(up(x_list.pop(), x_list.pop()))

        logits = self.outc(x_list[0])
        return logits

# # Version of UNet with Transformer in ending.

class UNetOutT(nn.Module):
    def __init__(self, n_channels, n_classes, unet_n_layers, start_scale, d_model, h, d_ff, num_layers, dropout):
        """
        :param n_channels: количество входных каналов
        :param n_classes: количество выходных каналов (классов)
        :param start_scale: с какого количества каналов начинает работать Unet
        :param d_model: количество ожидаемых features на входе трансформера
        :param h: количество heads в multiheadattention
        :param d_ff: количесвто нейроннов в полносвязной сети трансформера
        :param num_layers: количество слоёв в трансформере
        :param dropout:
        """
        super(UNetOutT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.unet_n_layers = unet_n_layers

        self.transformer = Transformer(d_model, h, d_ff, num_layers, n_classes, n_classes, dropout=dropout)

        self.inc = DoubleConv(n_channels, start_scale)

        if (unet_n_layers > 1):
            self.encoder = nn.ModuleList(
                [Down(start_scale * (2 ** (i - 1)), start_scale * (2 ** i)) for i in range(1, unet_n_layers)])
            self.decoder = nn.ModuleList(
                [Up(start_scale * (2 ** i), start_scale * (2 ** (i - 1))) for i in reversed(range(1, unet_n_layers))])

        self.outc = OutConv(start_scale, n_classes)

    def forward(self, x):
        x_list = [self.inc(x)]
        if self.unet_n_layers > 1:
            for down in self.encoder:
                x_list.append(down(x_list[-1]))
                
        if self.unet_n_layers > 1:
            for up in self.decoder:
                x_list.append(up(x_list.pop(), x_list.pop()))
        
        x_list.append(self.outc(x_list.pop()))
        x_list.append(self.transformer(x_list.pop().permute(0, 2, 1)))
        logits = x_list[-1]
        return logits

# # Draw function

def draw_prediction_and_reality(ecg_signal, prediction, right_answer, plot_name):
    """
    :param ecg_signal: сигнал некотего отведения
    :param prediction: предсказаные бинарные маски для этого отведения
    :param right_answer: правильная маска этого отведения (тоже три штуки)
    :param plot_name: имя картинки, куда хотим отрисовать это
    :return:
    """
    figname = plot_name + "_.png"
    print(ecg_signal.shape)
    print(prediction.shape)
    prediction = prediction.squeeze()
    #ecg_signal = ecg_signal.squeeze().cpu().detach().numpy()
    ecg_signal = ecg_signal.cpu().detach().numpy()
    right_answer = right_answer.squeeze().cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    #prediction = mm.fit_transform(prediction)

    print(right_answer.shape)
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=True, figsize=(20, 5))
    x = range(0, len(ecg_signal))
    #ecg_signal = ecg_signal.to('cpu')
    #right_answer = right_answer.to('cpu')
    #prediction = prediction.to('cpu')
    ax1.plot(ecg_signal[:,0], color='black')

    ax1.fill_between(x, 0, 100, where=right_answer[:, 0]>0.5, alpha=0.5, color='red')
    ax1.fill_between(x, 0, 100, where=right_answer[:, 1]>0.5, alpha=0.5, color='green')
    ax1.fill_between(x, 0, 100, where=right_answer[:, 2]>0.5, alpha=0.5, color='blue')

    ax1.fill_between(x, 120, 220, where=prediction[:, 0] > 0.5, alpha=0.8, color='red')
    ax1.fill_between(x, 120, 220, where=prediction[:, 1] > 0.5, alpha=0.8, color='green')
    ax1.fill_between(x, 120, 220, where=prediction[:, 2] > 0.5, alpha=0.8, color='blue')

    ax2.plot(prediction[:,0], 'r-')
    ax2.plot(prediction[:,1], 'g-')
    ax2.plot(prediction[:,2], 'b-')

    plt.legend(loc=2)
    plt.show()

def draw_prediction_and_reality_2(ecg_signal, prediction, right_answer, plot_name):
    """
    :param ecg_signal: сигнал некотего отведения
    :param prediction: предсказаные бинарные маски для этого отведения
    :param right_answer: правильная маска этого отведения (тоже три штуки)
    :param plot_name: имя картинки, куда хотим отрисовать это
    :return:
    """
    figname = plot_name + "_.png"
    print(ecg_signal.shape)
    print(prediction.shape)
    prediction = prediction.squeeze()
    #ecg_signal = ecg_signal.squeeze().cpu().detach().numpy()
    ecg_signal = ecg_signal.cpu().detach().numpy()
    right_answer = right_answer.squeeze().cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()

    print(right_answer.shape)
    f, ax1 = plt.subplots(1, 1, sharey=False, sharex=True, figsize=(15, 5))
    x = range(0, len(ecg_signal))
    #ecg_signal = ecg_signal.to('cpu')
    #right_answer = right_answer.to('cpu')
    #prediction = prediction.to('cpu')
    ax1.plot(ecg_signal[:,0], color='black')

    ax1.fill_between(x, 0, 100, where=right_answer[:, 0]>0.5, alpha=0.5, color='red', label='P волна')
    ax1.fill_between(x, 0, 100, where=right_answer[:, 1]>0.5, alpha=0.5, color='green', label='QRS комплекс')
    ax1.fill_between(x, 0, 100, where=right_answer[:, 2]>0.5, alpha=0.5, color='blue', label='T волна')

    ax1.fill_between(x, 120, 220, where=prediction[:, 0] > 0.5, alpha=0.8, color='red')
    ax1.fill_between(x, 120, 220, where=prediction[:, 1] > 0.5, alpha=0.8, color='green')
    ax1.fill_between(x, 120, 220, where=prediction[:, 2] > 0.5, alpha=0.8, color='blue')
    
    ax1.set_xlabel('Время (мс)')
    ax1.set_ylabel('Амплитуда (мВ)')

    plt.legend(loc=2)
    plt.savefig(plot_name + '.png')
    #plt.show()


# # Load Data

data = load_dataset()

X_train, X_test, Y_train, Y_test = train_test_split(data['x'], data['y'], test_size=0.33, random_state=42)


class LUBDDataset(torch.utils.data.Dataset):
    def __init__(self, signals, masks, names, win_len, num_leads_signal=12):
        self._signals = torch.FloatTensor(signals)
        self._masks = torch.LongTensor(masks)
        self._win_len = win_len
        self._num_leads_signal = num_leads_signal
        self.OFFSET = 700
    def __len__(self):
        return len(self._signals)

    def __getitem__(self, i):
        all_ecg_len = self._signals.shape[1]
        starting_position = np.random.randint(self.OFFSET, all_ecg_len - win_len - self.OFFSET)
        ending_position = starting_position + win_len
        return self._signals[i, starting_position:ending_position, 0:num_leads_signal].to(device), self._masks[i, starting_position:ending_position, :].to(device)

def get_loader(signals, masks, win_len, batch_size, num_leads_signal=12):
    dataset = LUBDDataset(signals, masks, win_len, num_leads_signal)
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size)


# +
# for description
# X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(data['x'], data['y'], data['text'], test_size=0.33, random_state=42)

# +
# class LUBDDataset(torch.utils.data.Dataset):
#     def __init__(self, signals, masks, names, win_len, num_leads_signal=12):
#         self._signals = torch.FloatTensor(signals)
#         self._masks = torch.LongTensor(masks)
#         self._names = names
#         self._win_len = win_len
#         self._num_leads_signal = num_leads_signal
#         self.OFFSET = 700
#     def __len__(self):
#         return len(self._signals)

#     def __getitem__(self, i):
#         all_ecg_len = self._signals.shape[1]
#         starting_position = np.random.randint(self.OFFSET, all_ecg_len - win_len - self.OFFSET)
#         ending_position = starting_position + win_len
#         return self._signals[i, starting_position:ending_position, 0:num_leads_signal].to(device), self._masks[i, starting_position:ending_position, :].to(device), self._names[i]

# +
# def get_loader(signals, masks, names, win_len, batch_size, num_leads_signal=12):
#     dataset = LUBDDataset(signals, masks, names, win_len, num_leads_signal)
#     from torch.utils.data import DataLoader
#     return DataLoader(dataset, batch_size=batch_size)
# -

win_len = 3072
batch_size = 4
num_leads_signal = 12

train_loader = get_loader(X_train, Y_train, win_len, batch_size, num_leads_signal)
val_loader = get_loader(X_test, Y_test, win_len, batch_size, num_leads_signal)

# +
# train_loader = get_loader(X_train, Y_train, Z_train, win_len, batch_size, num_leads_signal)
# val_loader = get_loader(X_test, Y_test, Z_test, win_len, batch_size, num_leads_signal)
# -


# # Activation functions and lose functions

# +
import torch.nn.functional as F

def get_softmax(logits):
    softmax = nn.Softmax(dim=2)
    res = logits.permute((0, 2, 1))
    res = softmax(res)
    return res.permute((0, 2, 1))

def get_ReLU(logits):
    relu = nn.ReLU()
    return relu(logits)

def mock_func_active(logits):
    return logits

def get_loss(model, X_batch, y_batch, func_active):
    X_batch = X_batch.permute((0, 2, 1))
    logits = model(X_batch)  # shape: (8, 4, 3072)
    logits = func_active(logits)
    #logits = logits[:, :, 1000:-1000]
    y_batch = torch.argmax(y_batch, dim=2)
    #y_batch = y_batch[:, 1000:-1000]
    #return F.cross_entropy(logits, y_batch).mean()
    return F.cross_entropy(logits, y_batch, torch.tensor([0.1, 0.7, 0.1, 0.1]).to(device)).mean()
    #return F.cross_entropy(logits, y_batch, torch.tensor([0.1,0.25,0.2,0.45]).to(device)).mean()
    #return F.cross_entropy(logits, y_batch, torch.tensor([0.1, 0.4, 0.3, 0.2]).to(device)).mean()

def get_bce_loss(model, X_batch, y_batch, func_active):
    X_batch = X_batch.permute((0, 2, 1))
    logits = model(X_batch)  # shape: (8, 4, 3072)
    logits = func_active(logits).permute((0, 2, 1))
    #logits = logits[:, :, 1000:-1000]
    #y_batch = torch.argmax(y_batch, dim=2)
    #y_batch = y_batch[:, 1000:-1000]
    return F.binary_cross_entropy_with_logits(logits.float(), y_batch.float()).mean()


# -

# # Drawer class

from IPython.display import clear_output
class Drawer():
    def __init__(self):
        self._train = []
        self._val = []

    def add(self, train_loss, val_iou):
        self._train.append(train_loss)
        self._val.append(val_iou)

    def plot(self):
        epochs = range(1, 1 + len(self._train))

        fig, ax = plt.subplots(1,1, figsize=(10, 8))
        ax.set_title('Train and test loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.plot(epochs, self._train, 'b--', label = "Train loss")
        ax.plot(epochs, self._val, c='red', label = "Test loss")
        ax.legend()
        #plt.savefig('loss.png')
        plt.show()
    def save_plot(self):
        epochs = range(1, 1 + len(self._train))

        fig, ax = plt.subplots(1,1, figsize=(10, 8))
        ax.set_title('Train and test loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.plot(epochs, self._train, 'b--', label = "Train loss")
        ax.plot(epochs, self._val, c='red', label = "Test loss")
        ax.legend()
        plt.savefig('loss.png')

min_losses = []


# # Fit function

def fit(model, train_loader, val_loader, optimizer, scheduler, loss_function, func_active, num_epochs=None):
    import time
    train_losses = []
    val_losses = []
    drawer = Drawer()
    min_loss = 1e6
    min_loss_index = 0
    epoch = 0

    while num_epochs is None or epoch < num_epochs:
        start_time = time.time()
        model.train()
        for X_batch, y_batch in train_loader:
            loss = loss_function(model, X_batch.to(device), y_batch.to(device), func_active)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().data.numpy())

        model.eval()
        for X_batch, y_batch in val_loader:
            with torch.no_grad():
                loss = loss_function(model, X_batch.to(device), y_batch.to(device), func_active)
                val_losses.append(loss.cpu().data.numpy())

        train_loss = np.mean(train_losses[-len(train_loader):])
        val_loss = np.mean(val_losses[-len(val_loader):])
        #print(len(train_loader), len(val_loader))
        drawer.add(train_loss, val_loss)
        clear_output(True)
        drawer.plot()
        print("Epoch {} took {:.3f}s".format(epoch + 1, time.time() - start_time))
        print("training loss: \t{:.6f}".format(train_loss))
        print("validation loss: \t{:.6f}".format(val_loss))

        if val_loss < min_loss - 1e-5:
            min_loss = val_loss
            min_loss_index = epoch
        elif epoch - min_loss_index >= 10:
            break

        epoch += 1
        scheduler.step()

    min_losses.append(min_loss)
    drawer.save_plot()

best_params = {
    "d_ff": 128,
    "d_model": 128,
    "dropout": 0.1,
    "h": 2,
    "num_layers": 6,
    "start_scale": 8,
    "unet_n_layers": 5
}


model = UNetMiddleT(12, 4, best_params['unet_n_layers'], best_params['start_scale'],
                    best_params['d_model'], best_params['h'], best_params['d_ff'],
                    best_params['num_layers'], best_params['dropout'])
model = model.to(device)

# +
# best_model = UNetMiddleT(n_channels=12, n_classes=4, unet_n_layers = 5, start_scale = 8, d_model = 128, h = 2, d_ff = 128, num_layers = 6, dropout=0.1)
# best_model = best_model.to(device)

# +
# unet = UNetV2(n_channels=12, n_classes=4, n_layers=5, start_scale=8)
# unet = unet.to(device)
# -

print(sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma=0.5)
num_epochs = 80
fit(model, train_loader, val_loader, optimizer, scheduler, get_loss, mock_func_active, num_epochs)

min_losses

# d_model: количество ожидаемых features на входах энкодера/декодера
#
# nhead: количество heads in the multiheadattention models
#
# d_hid: the dimension of the feedforward network model
#
# nlayers: the number of sub-encoder-layers in the encoder

model.eval()
with torch.no_grad():
    pred_test = model(torch.FloatTensor(X_test[:,1000:4072,:]).to(device).permute((0, 2, 1))).permute((0, 2, 1))
    #pred_test = model(torch.FloatTensor(X_test[:,:,:]).to(device)).permute((0, 2, 1))
    #pred_test = pred_test.permute((0, 2, 1))
    pred_test = F.one_hot(torch.argmax(pred_test, dim=2))
    pred_test = pred_test.cpu().detach().numpy()
print(statistics(Y_test[:, 1000:4072, :], pred_test[:, :, :]).round(4))

# +
#torch.save(model.state_dict(), r'out_t')
#model.load_state_dict(torch.load('123', map_location=device))
#model.load_state_dict(torch.load('ctt_unet_011_up'))
# -

model.load_state_dict(torch.load('save.pt', map_location=device))

# +
#best_model.load_state_dict(torch.load(r'C:\Users\game-\Desktop\Учёба\Python\PyThorch\Transformer\UNet_Middle_T\!\save.pt'))
# -

# # Plots

# +
it = iter(val_loader)
model.eval()
with torch.no_grad():
    x_test,y_test = next(it)
    #x_test,y_test,z_test = next(it)
    #x_test,y_test,z_test = next(it)
    predict = model(x_test.permute((0, 2, 1))).permute((0, 2, 1))
    #predict = model(x_test).permute((0, 2, 1))
    #predict = get_softmax(predict)
    predict = F.one_hot(torch.argmax(predict, dim=2))
    predict = preproc(predict)


k = 0
draw_prediction_and_reality(x_test[k,:,:], predict[k, :, :], y_test[k, :, :], "a")
# +
# save_path = r'C:\Users\game-\Desktop\Учёба\Python\PyThorch\Transformer\graphs1'
# -

"".join(z_test)

# +
# j = 0
# model.eval()
# for x_test, y_test, z_test in val_loader:
#     with torch.no_grad():
#         predict = model(x_test.permute((0, 2, 1))).permute((0, 2, 1))
#         predict = F.one_hot(torch.argmax(predict, dim=2))
#         predict = preproc(predict)
#     for i in range(len(x_test)):
#         with open(save_path + f'\\{j}_desc.txt', 'w') as f:
#             f.write(z_test[i])
#             draw_prediction_and_reality_2(x_test[i,:,:], predict[i, :, :], y_test[i, :, :], save_path + f"\\{j}")
#             j += 1
# -

it = iter(val_loader)

r = 1 #16

for _ in range(r):
    x_test,y_test,z_test = next(it)

j = 0
for x_test, y_test, z_test in val_loader:
    if j == r:
        break
    j += 1

model.eval()
with torch.no_grad():
    predict = model(x_test.permute((0, 2, 1))).permute((0, 2, 1))
    #predict = model(x_test).permute((0, 2, 1))
    #predict = get_softmax(predict)
    predict = F.one_hot(torch.argmax(predict, dim=2))
    predict = preproc(predict)

k = 2
print(z_test[k])
draw_prediction_and_reality_2(x_test[k,:,:], predict[k, :, :], y_test[k, :, :], "transformer")

unet.eval()
with torch.no_grad():
    unet_predict = unet(x_test.permute((0, 2, 1))).permute((0, 2, 1))
    #predict = model(x_test).permute((0, 2, 1))
    #predict = get_softmax(predict)
    unet_predict = F.one_hot(torch.argmax(unet_predict, dim=2))
    unet_predict = preproc(unet_predict)

print(z_test[k])
draw_prediction_and_reality_2(x_test[k,:,:], unet_predict[k, :, :], y_test[k, :, :], "unet")

best_model.eval()
with torch.no_grad():
    best_model_predict = best_model(x_test.permute((0, 2, 1))).permute((0, 2, 1))
    #predict = model(x_test).permute((0, 2, 1))
    #predict = get_softmax(predict)
    best_model_predict = F.one_hot(torch.argmax(best_model_predict, dim=2))
    best_model_predict = preproc(best_model_predict)

print(z_test[k])
draw_prediction_and_reality_2(x_test[k,:,:], best_model_predict[k, :, :], y_test[k, :, :], "best_model")

any_save_path = r'C:\Users\game-\Desktop\Учёба\Python\PyThorch\Transformer\comp'

j = 0
for x_test, y_test, z_test in val_loader:
    model.eval()
    with torch.no_grad():
        predict = model(x_test.permute((0, 2, 1))).permute((0, 2, 1))
        predict = F.one_hot(torch.argmax(predict, dim=2))
        predict = preproc(predict)
        
    unet.eval()
    with torch.no_grad():
        unet_predict = unet(x_test.permute((0, 2, 1))).permute((0, 2, 1))
        unet_predict = F.one_hot(torch.argmax(unet_predict, dim=2))
        unet_predict = preproc(unet_predict)
        
    best_model.eval()
    with torch.no_grad():
        best_model_predict = best_model(x_test.permute((0, 2, 1))).permute((0, 2, 1))
        best_model_predict = F.one_hot(torch.argmax(best_model_predict, dim=2))
        best_model_predict = preproc(best_model_predict)
    
    for i in range(len(x_test)):
        with open(any_save_path + f'\\{j}_desc.txt', 'w') as f:
            f.write(z_test[i])
            draw_prediction_and_reality_2(x_test[i,:,:], predict[i, :, :], y_test[i, :, :], any_save_path + f"\\{j}_model")
            draw_prediction_and_reality_2(x_test[i,:,:], unet_predict[i, :, :], y_test[i, :, :], any_save_path + f"\\{j}_unet")
            draw_prediction_and_reality_2(x_test[i,:,:], best_model_predict[i, :, :], y_test[i, :, :], any_save_path + f"\\{j}_best")
            j += 1



# # Tune

from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler


def train_model_unet(model, optimizer, train_loader_):
    model.train()
    for X_batch, y_batch, _ in train_loader_:
        loss = get_loss(model, X_batch.to(device), y_batch.to(device), mock_func_active)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def test_model_unet(model, val_loader_):
    model.eval()
    for X_batch, y_batch, _ in val_loader_:
        with torch.no_grad():
            loss = get_loss(model, X_batch.to(device), y_batch.to(device), mock_func_active)
    return loss


def get_table(model, data_x_test, data_y_test, dir_path):
    model.eval()
    with torch.no_grad():
        pred_test = model(torch.FloatTensor(data_x_test[:,1000:4072,:]).to(device).permute((0, 2, 1))).permute((0, 2, 1))
        pred_test = F.one_hot(torch.argmax(pred_test, dim=2))
        pred_test = pred_test.cpu().detach().numpy()
    
    statistics(data_y_test[:, 1000:4072, :], pred_test[:, :, :]).round(4).to_csv(dir_path + '/table.csv')



import os
from ray.air.checkpoint import Checkpoint
def train_unet_middle_t(config, data):
    
    unet_n_layers = int(config['unet_n_layers'])
    start_scale = int(config['start_scale'])
    d_model = config['d_model']
    h = config['h']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    dropout = config['dropout']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dir_path = f"D:/Another_path/UNet_Middle_T/UNet_{unet_n_layers}_{start_scale}_{d_model}_{h}_{d_ff}_{num_layers}_{dropout}"
    os.makedirs(dir_path, exist_ok=True)
    loaded_checkpoint = session.get_checkpoint()

    
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(data['x'], data['y'], data['text'], test_size=0.33, random_state=42)
    
    train_loader = get_loader(X_train, Y_train, Z_train, win_len, batch_size, num_leads_signal)
    val_loader = get_loader(X_test, Y_test, Z_test, win_len, batch_size, num_leads_signal)
    
    num_epochs = 80
    
    model = UNetMiddleT(12,4, unet_n_layers, start_scale, d_model, h, d_ff, num_layers, dropout)
    model.to(device)
    
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"), map_location=device)
        model.load_state_dict(model_state)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    epoch = 0
    while num_epochs is None or epoch < num_epochs:
        
        loss = train_model_unet(model, optimizer, train_loader)
        train_losses.append(loss.cpu().data.numpy())

        loss = test_model_unet(model, val_loader)
        val_losses.append(loss.cpu().data.numpy())

        train_loss = np.mean(train_losses[-len(train_loader):])
        val_loss = np.mean(val_losses[-len(val_loader):])
        
        
        epoch += 1

        torch.save(model.state_dict(), dir_path + "/checkpoint.pt", _use_new_zipfile_serialization=False)
        checkpoint = Checkpoint.from_directory(dir_path)
        session.report({'train_loss': train_loss, "val_loss": val_loss}, checkpoint=checkpoint)
        get_table(model, X_test, Y_test, dir_path)

from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
def tune_unet_middle_t_opt_BOHB():
#     search_space = {
#         "unet_n_layers": tune.uniform(1,6),
#         "start_scale": tune.uniform(1,17)
#     }
    search_space = {
        "unet_n_layers": tune.choice([i for i in range(1,6)]),
        "start_scale": tune.choice([2 ** i for i in range(5)]),
        "d_model": tune.choice([2 ** i for i in range(4,10)]),
        "h": tune.choice([2**i for i in range(3)]),
        "d_ff": tune.choice([2 ** i for i in range(6,12)]),
        "num_layers": tune.choice([i for i in range(1,7)]),
        "dropout": tune.uniform(0.05,0.55)
    }
    alg = TuneBOHB(metric="val_loss", mode="min")
    scheduler = HyperBandForBOHB(max_t=80,metric="val_loss", mode="min")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_unet_middle_t, data=data),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            search_alg=alg,
            scheduler=scheduler,
            num_samples=1000
        ),
        param_space=search_space,
        run_config=air.RunConfig(
            name="tune_unet_middle_t",
            local_dir="D:/Another_path/ray_log"
        ),
        
    )
  
    results = tuner.fit()
    print(results.get_best_result("val_loss", mode="min"))
    results.get_dataframe().to_csv("D:/Another_path/UNet_Middle_T/res.csv")


tune_unet_middle_t_opt_BOHB()

from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
import ray
def tune_unet_middle_t_opt_BOHB_restore():
    
    search_space = {
    "unet_n_layers": tune.choice([i for i in range(1,6)]),
    "start_scale": tune.choice([2 ** i for i in range(5)]),
    "d_model": tune.choice([2 ** i for i in range(4,10)]),
    "h": tune.choice([2**i for i in range(3)]),
    "d_ff": tune.choice([2 ** i for i in range(6,12)]),
    "num_layers": tune.choice([i for i in range(1,7)]),
    "dropout": tune.uniform(0.05,0.55)
    }
    
    alg = TuneBOHB(metric="train_loss", mode="min")
    alg.restore_from_dir("D:/Another_path/ray_log/tune_unet_middle_t/")
    scheduler = HyperBandForBOHB(max_t=80,metric="train_loss", mode="min")
    ray.init()
    
    params = tune.with_resources(
             tune.with_parameters(train_unet_middle_t, data=data),
             resources={"cpu": 4, "gpu": 1}
             )
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_unet_middle_t, data=data),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            search_alg=alg,
            scheduler=scheduler,
            num_samples=100
        ),
        param_space=search_space,
        run_config=air.RunConfig(
            name="tune_unet_middle_t",
            local_dir="D:/Another_path/ray_log"
        ),
        
    )
    tuner = tuner.restore("D:/Another_path/ray_log/tune_unet_middle_t/", trainable=params, resume_errored=True)
    results = tuner.fit()
    print(results.get_best_result("val_loss", mode="min"))
    results.get_dataframe().to_csv("D:/Another_path/UNet_Middle_T/res.csv")
