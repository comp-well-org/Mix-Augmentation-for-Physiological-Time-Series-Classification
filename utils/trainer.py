import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import wandb 
from utils.augmentation import mixup_data, mixup_layer, cutmix, mixup_loss, cutout


def val_model(model, dataloaders, criterion, device, verbose, args):
    """
    Model evaluation.
    """
    test_losses = []
    test_pred_labels = []
    test_true_labels = []
    model.eval()
    for x, labels in dataloaders['test']:
        x = x.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        logits = model(x)
        if args.mixup and args.mixup_type == 'layer':
            logits = model.avgpool(logits)
            logits = torch.flatten(logits, 1)
            logits = model.fc(logits)
        loss = criterion(logits,labels)

        test_losses.append(loss.item())
        test_true_labels.append(labels.detach().cpu())
        test_pred_labels.append(logits.detach().cpu())

    all_pred = np.vstack(test_pred_labels)
    all_true = np.vstack(test_true_labels)

    all_pred_binary = np.argmax(all_pred, axis=1)
    all_true_binary = np.argmax(all_true, axis=1)
    val_acc = accuracy_score(all_true_binary, all_pred_binary)
    #all_pred_binary = logits_2_binary(all_pred)
    #np.save('./GSR.npy', all_pred_binary)
    #all_true_binary = all_true
    
    if verbose:
        print("\nValidataion:")
        print("Loss: %.4f" %(np.mean(np.array(test_losses))))
        #print("F1 score: %.4f" %(f1_score(all_true_binary, all_pred_binary)))
        print("Accuracy: %.4f " % val_acc)
        # print(confusion_matrix(all_true_binary, all_pred_binary))
    else:
        print("%.4f" %(np.mean(np.array(test_losses))), end = '\t')
        print("%.4f " % val_acc)
    return val_acc, all_true_binary, all_pred_binary


def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, args):

    wandb.init(project="Epilepsy", group="Jan", entity="pkglimmer", name=args.run_name)
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size
    }

    best_acc, best_epoch = -1, -1
    verbose = False
    mixup_func = {'data':mixup_data, 'cutmix':cutmix}

    print("Epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc")

    for epoch in range(args.num_epochs):
        if verbose:
            print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []
        correct, total = 0, 0 # temporary counter, for mixup only

        model.train()
        for x, labels in dataloaders['train']:
            # move data to GPU
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            # reset optimizer.
            optimizer.zero_grad()

            if args.mixup:
                if args.mixup_type == 'layer':
                    layer_out = model(x)
                    indices_perm, y_a, y_b, lam = mixup_layer(labels, device)
                    out = layer_out * lam + layer_out[indices_perm, :] * (1 - lam)
                    out = model.avgpool(out)
                    out = torch.flatten(out, 1)
                    logits = model.fc(out)
                    loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                else:
                    x, y_a, y_b, lam = mixup_func[args.mixup_type](x, labels, args.mixup_alpha)
                    x, y_a, y_b = map(Variable, (x, y_a, y_b))
                    logits = model(x)
                    # loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                train_losses.append(loss.item())
                _, predicted = torch.max(logits.data, 1)

                total += x.size()[0]
                # one-hot to numerical
                y_a = torch.argmax(y_a, axis=1)
                y_b = torch.argmax(y_b, axis=1)
                correct += (lam * predicted.eq(y_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(y_b.data).cpu().sum().float())
                train_acc = correct / total
            else:
                if args.cutout:
                    x = cutout(x)
                logits = model(x)
                labels = nn.functional.one_hot(labels.to(torch.int64), num_classes=2)
                loss = criterion(logits, labels.float())
                # obtain necessary information for displaying.
                train_losses.append(loss.item())
                train_pred_labels.append(logits.detach().cpu())
                train_true_labels.append(labels.detach().cpu())
                
                all_pred = np.vstack(train_pred_labels)
                all_true = np.vstack(train_true_labels)
                # convert from one-hot coding to binary label.
                all_pred_binary = np.argmax(all_pred, axis=1)
                all_true_binary = np.argmax(all_true, axis=1)
                train_acc = accuracy_score(all_true_binary, all_pred_binary)

            loss.backward()
            optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()
            
        # Show training info at the end of each epoch
        if verbose:
            print("Epoch {}/{}".format(epoch, args.num_epochs-1));
            print("Loss: %.4f" %(np.mean(np.array(train_losses))))
            #F1 = f1_score(all_true_binary, all_pred_binary)
            #print("F1 score: %.4f" %(F1))
            print("Train accuracy: %.4f " %(train_acc))
            if not args.mixup:
                print(confusion_matrix(all_true_binary, all_pred_binary))
        else:
            print(f"{epoch}", end = '\t');
            print("%.4f" %(np.mean(np.array(train_losses))), end = '\t')
            print("%.4f " %(train_acc), end = '\t')

        # validation
        val_acc, y_gt, y_pred = val_model(model, dataloaders, criterion, device, verbose, args)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            # if verbose:
            print("Saved new best model")
            wandb.run.summary["best_accuracy"] = best_acc
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_F1"] = f1_score(y_gt, y_pred, average='macro')
            mat = confusion_matrix(y_gt, y_pred)
            df_cm = pd.DataFrame(mat)
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', xticklabels=args.class_names, yticklabels=args.class_names)
            wandb.log({'confusion matrix': wandb.Image(plt)})
            plt.clf()
            
            torch.save(model.state_dict(), os.path.join(args.model_weights_dir, f'{args.run_name}_best.pth'))
        wandb.log({
                    "loss": np.mean(np.array(train_losses)),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "conf_mat":wandb.plot.confusion_matrix(probs=None,
                        y_true=y_gt, preds=y_pred,
                        class_names=args.class_names)
                })

    print(f'Training complete. Best validation accuracy {best_acc:.4f} at epoch {best_epoch}')    
    # torch.save(model.state_dict(), os.path.join(args.model_weights_dir, f'{args.run_name}_final.pth'))
    wandb.finish()



