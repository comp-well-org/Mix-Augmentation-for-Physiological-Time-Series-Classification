import os
import argparse
from time import time
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from models.model import *
from models.resnet1d import *
from utils.ptb_data import *
from config import PTB_XL_CONFIG as config

class T_sne_visual():
    def __init__(self, model, dataset, dataloader, fig_path, fig_title):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.save_path = fig_path
        self.fig_title = fig_title
        # self.class_list = dataset.classes
    def visual_dataset(self):
        imgs = []
        labels = []
        for img, label in self.dataset:
            imgs.append(np.array(img).transpose((2, 1, 0)).reshape(-1))
            # tag = self.class_list[label]
            labels.append(label)
        self.t_sne(np.array(imgs), labels,title=f'Dataset visualize result\n')

    def visual_feature_map(self, layer):
        self.model.eval()
        with torch.no_grad():
            self.feature_map_list = []
            labels = []
            getattr(self.model, layer).register_forward_hook(self.forward_hook)
            for img, label in self.dataloader:
                img=img.to('cuda:0', dtype=torch.float32)
                self.model(img)
                for i in label.tolist():
                    # tag=self.class_list[i]
                    labels.append(np.argmax(i))
            self.feature_map_list = torch.cat(self.feature_map_list, dim=0)
            self.feature_map_list = torch.flatten(self.feature_map_list, start_dim=1)
            self.t_sne(np.array(self.feature_map_list.cpu()), np.array(labels), title=f'{self.fig_title} feature tSNE\n', save_path=self.save_path)

    def forward_hook(self, model, input, output):
        self.feature_map_list.append(output)

    def set_plt(self, start_time, end_time,title):
        plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
        plt.legend(title='')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def t_sne(self, data, label, title, save_path):
        print('starting T-SNE process')
        start_time = time()
        data = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y']) 
        df.insert(loc=1, column='label', value=label)
        end_time = time()
        print('Finished')

        # plot
        sns.scatterplot(x='x', y='y', hue='label', s=3, palette="Set2", data=df)
        self.set_plt(start_time, end_time, title)
        plt.savefig(save_path, dpi=400)
        plt.show()


base_path = config.BASE_PATH
train_test_str = 'train'
test_df = pd.read_csv(os.path.join(base_path, f'{train_test_str}_meta.csv'), index_col=0)
test_signal = pd.read_csv(os.path.join(base_path, f'{train_test_str}_signal.csv'))
data_preprocessor = PTBXLDatasetPreprocesserSuperclassOnly()
test_signal, test_superclass = data_preprocessor.transform(test_signal, test_df) 
dataset = ECGDatasetSuperclassesOnly(test_signal, test_superclass)
TS_dataloader = torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle=False)
model = resnet18(num_input_channels=12, num_classes=5)
ckpt = 'checkpoints/resnet18_AdamW_0.005_vanilla__best.pth' #





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',  type=int, default=0, help="ckpt index")
    parser.add_argument('--note', type=str, default='')
    args = parser.parse_args()
    
    model.load_state_dict(torch.load(ckpt))
    # print(model)
    model.cuda()

    save_path = './imgs'
    title = 'PTB-XL ' + '_'.join(ckpt.split('/')[-1].split('_')[3:-1])
    note_str = args.note
    save_path = os.path.join(save_path, title) + f'{note_str}.png'
    title += f' {note_str}'
    print(f'save path {save_path}')
    t = 0
    t = T_sne_visual(model, dataset, TS_dataloader, save_path, title)
    # t.visual_dataset()
    t.visual_feature_map('layer4')
    