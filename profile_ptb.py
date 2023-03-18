import os
from config import PTB_XL_CONFIG
from main import run

archs = ['resnet18', 'resnet50']
mixup_types = ['data', 'cutmix', 'layer']
mixup_flag = [True, False]
mixup_alphas = [0.4, 0.75]
balanced_samplers = [True, False]
optimizers = ['Adam', 'AdamW']
lrs = [5e-3, 1e-3]

# sweep params

config = PTB_XL_CONFIG()

for arch in archs:
    for lr in lrs:
        for optimizer in optimizers:
            for balanced_sampler in balanced_samplers:
                config.arch = arch 
                config.lr = lr
                config.optimizer = optimizer 
                config.balanced_sampler = balanced_sampler
                # no mixup
                for mixup in mixup_flag:
                    if mixup:
                        config.mixup = True 
                        for mixup_type in mixup_types:
                            for mixup_alpha in mixup_alphas:
                                config.mixup_type = mixup_type
                                config.mixup_alpha = mixup_alpha
                                da_str = f'mixup({mixup_type},{mixup_alpha})'
                                balanced_str = 'balanced' if balanced_sampler else ''
                                config.run_name = f'{arch}_{optimizer}_{lr}_{da_str}_{balanced_str}'
                                run(config)
                    else:
                        config.mixup = False 
                        balanced_str = 'balanced' if balanced_sampler else ''
                        config.run_name = f'{arch}_{optimizer}_{lr}_vanilla_{balanced_str}'
                        run(config)
                
