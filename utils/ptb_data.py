import torch
import os
import random
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings; warnings.filterwarnings('ignore')
from models.model import ECGDataset, ECGDatasetSuperclassesOnly


# preprocessing
class PTBXLDatasetPreprocesser():
    def __init__(self):
        pass
    
    def save(self, filename):
        data = {
            'superclass_cols': self.superclass_cols,
            'subclass_cols': self.subclass_cols,
            'meta_num_cols': self.meta_num_cols,
            'meta_num_means': self.meta_num_means,
            'min_max_scaler': self.min_max_scaler,
            'meta_cat_cols': self.meta_cat_cols,
            'cat_lablers': self.cat_lablers,
        }
        pd.to_pickle(data, filename)
        
    def load(self, filename):
        data = pd.read_pickle(filename)
        self.min_max_scaler = data['min_max_scaler']
        self.cat_lablers = data['cat_lablers']
        
    def fit(self, x, y):
        x = x.copy()
        y = y.copy()
        
        self.superclass_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        self.subclass_cols = [col for col in y.columns if 'sub_' in col]
        
        self.meta_num_cols = ['age', 'height', 'weight']
        self.meta_num_means = []
        for col in self.meta_num_cols:
            print(col, y[col].mean())
            y[col] = y[col].fillna(y[col].mean())
            self.meta_num_means += [y[col].mean()]
            
        self.min_max_scaler = MinMaxScaler().fit(y[self.meta_num_cols])
        
        self.meta_cat_cols = ['sex'] #, 'nurse', 'device']
        self.cat_lablers = [LabelEncoder().fit(y[col].fillna('none').astype(str)) for col in self.meta_cat_cols]
        return self
    
    def transform(self, x, y):
        
        channel_cols = x.columns.tolist()[1:]
        
        ret = []
        x = x[channel_cols].values.reshape(-1, 1000, 12)
        print(x.shape)
        ret += [x] # signal
        
        y_ = y.copy()
        
        for i, col in enumerate(self.meta_num_cols):
            y_[col] = y_[col].fillna(self.meta_num_means[i])
        y_[self.meta_num_cols] = self.min_max_scaler.transform(y_[self.meta_num_cols])
        y_[self.meta_num_cols] = np.clip(y_[self.meta_num_cols], 0., 1.) # prevent extreme value far from train set
        
        ret += [y_[self.meta_num_cols]] # meta num features
        
        for i, col in enumerate(self.meta_cat_cols):
            y_[col] = y_[col].fillna('none').astype(str)
            y_[col] = self.cat_lablers[i].transform(y_[col]) 
        
        ret += [y_[self.meta_cat_cols]] # meta cat features
        
        if np.isin(self.superclass_cols, y.columns).sum() == len(self.superclass_cols):
            ret += [y[self.superclass_cols].fillna(0).astype(int)] # superclass targets
        
        if np.isin(self.subclass_cols, y.columns).sum() == len(self.subclass_cols):
            ret += [y[self.subclass_cols].fillna(0).astype(int)] # subclass targets
        
        return ret
    
class PTBXLDatasetPreprocesserSuperclassOnly():
    def __init__(self):
        self.superclass_cols = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    def transform(self, x, y):
        
        channel_cols = x.columns.tolist()[1:]
        ret = []
        x = x[channel_cols].values.reshape(-1, 1000, 12)
        single_label_indices = y[self.superclass_cols].sum(axis=1) == 1
        y = y[self.superclass_cols][single_label_indices]
        x = x[single_label_indices]
        
        return [x, y]
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def prepare_dataloader(signal, meta_num_feats, meta_cat_feats, superclass, subclass):
    
    ds = ECGDataset(signal, meta_num_feats, meta_cat_feats, superclass_labels=superclass, subclass_labels=subclass)
    
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=128,
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=4,
        #sampler=BalanceClassSampler(labels=y_train, mode='downsampling'),
    )
    return dl

def prepare_dataloader_superclass_only(signal, superclass, batch_size):
    
    dataset = ECGDatasetSuperclassesOnly(signal, superclass_labels=superclass)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=4,
        #sampler=BalanceClassSampler(labels=y_train, mode='downsampling'),
    )
    return dataloader

