import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd 
import os

class ECGDataset(Dataset):
    def __init__(self, signals, num_metas, cat_metas, superclass_labels=None, subclass_labels=None):
        self.signals = signals
        self.num_metas = num_metas
        self.cat_metas = cat_metas
        self.superclass_labels = superclass_labels
        self.subclass_labels = subclass_labels
        
    def __len__(self):
        return self.signals.shape[0]
    
    def __getitem__(self, idx):
        
        ret = []
        ret += [self.signals[idx,:]]
        ret += [self.num_metas.values[idx,:]]
        ret += [self.cat_metas.values[idx,:]]
        
        if self.superclass_labels is not None:
            ret += [self.superclass_labels.values[idx,:]]
        
        if self.subclass_labels is not None:
            ret += [self.subclass_labels.values[idx,:]]
        
        return ret


class ECGDatasetSuperclassesOnly(Dataset):
    def __init__(self, signals, superclass_labels):
        self.signals = signals
        self.superclass_labels = superclass_labels
        
    def __len__(self):
        return self.signals.shape[0]
    
    def __getitem__(self, idx):
        return [self.signals[idx, :], self.superclass_labels.values[idx, :]]
    
    def get_labels(self):
        labels = np.argmax(self.superclass_labels.values, 1)
        return labels
    
DATA_DIMS = [121, 129 * 9]

class MsgDataset(Dataset):
    def __init__(self):
        self.base_path = '/rdf/data/msg2022/preprocessed/train/'
        data = pd.read_csv('/rdf/data/msg2022/train/train_labels.csv')
        self.labels = data['label']
        self.paths = data['filepath']
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_path, self.paths[idx])
        path = path.replace('parquet', 'bin')
        x = torch.Tensor(np.fromfile(path)).reshape(DATA_DIMS[::-1]) # TODO
        label = self.labels[idx]
        return x, label
    
class ECGClassifier(nn.Module):
    def __init__(self, signal_channel_size, gru_hidden_size, per_cat_nunique, embed_size, num_size, hidden, n_outs):
        super().__init__()
        
        self.gru1 = nn.GRU(signal_channel_size, gru_hidden_size, batch_first=True, bidirectional=True)
        #self.lstm2 = nn.LSTM(gru_hidden_size*2, gru_hidden_size, batch_first=True, bidirectional=True)
        
        self.embeds = []
        self.per_cat_nunique = per_cat_nunique
        for v in self.per_cat_nunique:
            self.embeds += [nn.Embedding(v, embed_size)]
        self.embeds = nn.ModuleList(self.embeds)
        
        self.dense1 = nn.Linear(gru_hidden_size*4 + embed_size*len(per_cat_nunique) + num_size, hidden)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden, n_outs)
        
    def forward(self, signal, num_meta, cat_meta):
        #print(signal.shape)
        signal = signal.view(signal.shape[0], signal.shape[1], -1)
        #print(signal.shape)
        signal, _ = self.gru1(signal)
        #signal, _ = self.lstm2(signal)
        
        avg_pool = torch.mean(signal, 1)
        max_pool, _ = torch.max(signal, 1)
        
        cat_feats = []
        for i, embed in enumerate(self.embeds):
            cat_feats += [embed(cat_meta[:,i].long())]
        cat_feats = torch.cat(cat_feats, 1) 
        
        x = torch.cat([avg_pool, max_pool, cat_feats, num_meta], 1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.out(x)
        
        return x
    
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x;

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class model_ResNet(nn.Module):
    """
        layers = [2,2,2,2] by default (ResNet18)
        [3,4,6,3] ResNet34
    """
    def __init__(self, layers=[2, 2, 2, 2], block=BasicBlock, num_classes=5, dropout_rate=0.5, is_training=True, layer_mixup=False):
        super(model_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=3, stride=1, padding=1) # 12-lead input
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.avgpool = nn.AvgPool1d(kernel_size=47)  # TODO
        self.fc = nn.Linear(512, num_classes)   # the value is undecided yet.
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_mixup = layer_mixup

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None;
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes*block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride,  downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        
        return  nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2) # dimensions of dim-1 and dim 2 are swapped. [B, Length, C] => [B, C, Length]
        x = self.conv1(x);
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.layer_mixup and self.training:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

# to define a resnet model
#  model = model_ResNet(BasicBlock, [2, 2, 2, 2])

def main():
    # verify the models with dummy tensor
    model = model_ResNet(layers=[2,2,2,2], num_classes=5)
    dummy = torch.randn((10, 1000, 12)) # PTB-XL
    print(f'dummy.shape {dummy.shape}')
    x = model(dummy)
    print(f'x.shape {x.shape}')
    
if __name__ == "__main__":
    main()
