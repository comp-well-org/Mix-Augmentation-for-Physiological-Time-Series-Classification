import yaml
from re import L
import wandb
import sys
from utils.ptb_data import *
from utils.ptb_trainer import *
from utils.trainer import train_model
from models.model import model_ResNet, MsgDataset
from models.resnet1d import *
from torch.cuda.amp.grad_scaler import GradScaler
from config import PTB_XL_CONFIG
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data.sampler import SubsetRandomSampler

models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'x50': resnext50_32x4d,
    'x101': resnext101_32x8d
}

optimizers = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW
}


def run(config = None):
    # for training only, need nightly build pytorch
    config = PTB_XL_CONFIG() if config is None else config
    
    print('start testing..............')
    attrs = vars(config)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('='*20)
        
    base_path = config.BASE_PATH
    train_df = pd.read_csv(os.path.join(base_path, 'train_meta.csv'), index_col=0)
    train_signal = pd.read_csv(os.path.join(base_path, 'train_signal.csv'))

    valid_df = pd.read_csv(os.path.join(base_path, 'valid_meta.csv'), index_col=0)
    valid_signal = pd.read_csv(os.path.join(base_path, 'valid_signal.csv'))

    test_df = pd.read_csv(os.path.join(base_path, 'test_meta.csv'), index_col=0)
    test_signal = pd.read_csv(os.path.join(base_path, 'test_signal.csv'))

    data_preprocessor = PTBXLDatasetPreprocesserSuperclassOnly()
    train_signal, train_superclass = data_preprocessor.transform(train_signal, train_df)  # 12961 / 17420 single-label
    valid_signal, valid_superclass = data_preprocessor.transform(valid_signal, valid_df) # 1637 / 2183 single-label
    test_signal, test_superclass = data_preprocessor.transform(test_signal, test_df) # 1650 / 2198 single-label

    seed = 2022
    batch_size = 256
    stepsize= 8
    seed_everything(seed)
    
    train_loader = prepare_dataloader_superclass_only(train_signal, train_superclass, batch_size)
    val_loader = prepare_dataloader_superclass_only(valid_signal, valid_superclass, batch_size)
    test_loader = prepare_dataloader_superclass_only(test_signal, test_superclass, batch_size)

    dataset = ECGDatasetSuperclassesOnly(train_signal, train_superclass)
    if config.balanced_sampler:
        sampler = ImbalancedDatasetSampler(dataset)
        train_loader = torch.utils.data.DataLoader(dataset, sampler = sampler, batch_size = config.batch_size)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size = config.batch_size, shuffle=True)

    device = torch.device('cuda:0')
    layer_mixup = config.mixup == True and config.mixup_type == 'layer'

    model = models[config.arch](num_input_channels=12, num_classes=config.num_classes, layer_mixup=layer_mixup).to(device)
            
    optimizer = optimizers[config.optimizer](model.parameters(), config.lr, weight_decay=0.009)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=config.lr_stepsize) if config.lr_scheduler else None

    # criterion = nn.CrossEntropyLoss().to(device) # for multi-label/mixup 
    criterion = nn.BCEWithLogitsLoss.to(device) # For single-label classification
    

    # for epoch in range(epochs):
    #     train_one_epoch(epoch, model, criterion, optimizer, train_loader, device, scaler=scaler, scheduler=scheduler, schd_batch_update=False)

    #     with torch.no_grad():
    #         val_targets, val_preds = valid_one_epoch(epoch, model, val_loader, device)
    #         test_targets, test_preds = valid_one_epoch(epoch, model, test_loader, device)
    
    dataloaders = {'train': train_loader, 'test': test_loader, 'val': val_loader}
    train_model(model, dataloaders, criterion, optimizer, lr_scheduler=scheduler, device=device, args=config)
    # torch.save(model.state_dict(),'model_final.pth')



if __name__ == '__main__':
    run()
