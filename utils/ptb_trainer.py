import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score
from utils.augmentation import mixup_data, cutmix
import warnings; warnings.filterwarnings('ignore')

def train_one_epoch(epoch, model, criterion, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    data_mixup = True

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (signal, superclass_labels) in pbar:
        signal = signal.to(device).float() # shape: [batch, 1000, 12]
        labels = superclass_labels.to(device).float()
            
        with autocast():
            if data_mixup:
                # mixed_signal, y_a, y_b, lam = mixup_data(signal, labels, 0.4)
                mixed_signal, y_a, y_b, lam = cutmix(signal, labels)
                preds = model(mixed_signal)
                loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
            else:
                preds = model(signal)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()

            running_loss = loss.item()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad() 

        if scheduler is not None and schd_batch_update:
            scheduler.step()
            
        description = f'epoch {epoch} loss: {running_loss:.4f}'
        pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
def valid_one_epoch(epoch, model, val_loader, device):
    model.eval()

    loss_sum = 0
    sample_num = 0
    preds_all = []
    targets_all = []
    
    for _, (signal, superclass_labels) in enumerate(val_loader):
        signal = signal.to(device).float()
        
        labels = superclass_labels.to(device).float()
        preds = model(signal)
        preds = preds.softmax(1)
    
        
        preds_all += [preds.detach().cpu().numpy()]
        targets_all += [labels.detach().cpu().numpy()]
    
        sample_num += labels.shape[0]  

    
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    print('validation superclass single-label acc = {:.4f}'.format((np.argmax(targets_all, 1)==np.argmax(preds_all, 1)).mean()))
    print('validation superclass single-label macro auc = {:.4f}'.format(roc_auc_score(targets_all, preds_all, average='macro')))
            
    return targets_all, preds_all