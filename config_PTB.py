# Configuration file for training on PTB-XL dataset

class PTB_XL_CONFIG:
    BASE_PATH = '[PATH_TO_YOUR_PTBXL_DATASET]'
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    optimizer = 'AdamW'
    lr = 1e-2
    num_epochs = 50
    lr_scheduler = True
    lr_stepsize = 5
    batch_size = 128
    num_classes = 5
    balanced_sampler = False
    arch = 'resnet18'
    mixup = True
    mixup_type = 'layer' # 'data' or 'cutmix'
    mixup_alpha = 0.4
    da_str = 'vanilla' if not mixup else f'mixup({mixup_type},{mixup_alpha})'
    balanced_str = 'balanced' if balanced_sampler else ''
    model_weights_dir = '[PATH_TO_CHECKPOINTS]'
    run_name = f'{arch}_{optimizer}_{lr}_{da_str}_{balanced_str}'
    
    
