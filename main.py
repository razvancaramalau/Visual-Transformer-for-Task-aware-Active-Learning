'''
Visual Transformer for Task-aware Active Learning
'''
# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
import yaml
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet, GCN, Discriminator1, Discriminator2
from train_test import train, test, train_with_sampler, test_with_sampler
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *
import sys
sys.path.append(".")


parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", type=str, default="cifar10",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=220, 
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="JLS",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-ws","--which_synthetic", type=str, default="stargan",
                    help="")
parser.add_argument("-v","--visual_transformer", type=bool, default=True,
                    help="")

args = parser.parse_args()

##
# Main
if __name__ == '__main__':
    drop_flag = args.visual_transformer
    method = args.method_type
    methods = ['Random', 'CoreSet', 'lloss','VAAL',
                'CDAL','JLS','TJLS']
    datasets = ['cifar10', 'cifar100', 'fashionmnist','svhn','rafd']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    '''
    method_type: 'Random', 'CoreSet', 'lloss','VAAL', 'JLS', 'TJLS', 'CDAL'
    '''
    # with open('config.yaml','r') as config_file:
    #     config = yaml.load(config_file, Loader=yaml.FullLoader)
    #lr1e-2_interl_SGD0.5_d2_
    results = open('results__'+str(args.method_type)+"_"+args.dataset +'_main'+str(args.cycles)+
                    str(args.total)+str(args.visual_transformer)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles
    if (args.method_type == "JLS") or (args.method_type == "TJLS"):
        args.visual_transformer = True
    else:
        args.visual_transformer = False
    for trial in range(TRIALS):

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        ADDENDUM = ADDENDUM
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        init_margin = int(NUM_TRAIN/10)
        if args.total or (args.dataset=='rafd'):
            labeled_set= indices
            unlabeled_set = [x for x in range(0, NUM_TRAIN)]
        else:
            # take 10% of the labelled data at first run
            # if os.path.isfile("init_set.npy"):
            #     labeled_set = np.load("init_set.npy").tolist()
            # else:
            labeled_set = indices[:init_margin] 
                # np.save("init_set.npy", np.asarray(labeled_set))

            # if os.path.isfile("init_unl_set.npy"):
            #     unlabeled_set = np.load("init_unl_set.npy").tolist()
            # else:
            unlabeled_set = [x for x in indices if x not in labeled_set]
                # unlabeled_set = unlabeled_set[:5000]
                # np.save("init_unl_set.npy", np.asarray(unlabeled_set))
        synth_set = []
        train_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetRandomSampler(labeled_set), 
                                    pin_memory=True, drop_last=drop_flag)


        test_loader  = DataLoader(data_test, batch_size=BATCH, drop_last=drop_flag)
        dataloaders  = {'train': train_loader, 'test': test_loader}


        for cycle in range(7):
            
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set
                if (args.method_type == "JLS") or (args.method_type == "TJLS"):
                    if args.dataset=='rafd':
                        subset = unlabeled_set[:1000]
                        
                        if cycle>0:
                            labeled_set = concat_set
                        interleaved_size = 2 * int(len(subset)/BATCH) * BATCH
                        subset = [x + 7200 for x in subset]
                    else:
                        subset = unlabeled_set
                        if len(subset)>len(labeled_set):
                            interleaved_size = 2 * int(len(labeled_set)/BATCH) * BATCH
                        else:
                            interleaved_size = 2 * int(len(subset)/BATCH) * BATCH


                    interleaved = np.zeros((interleaved_size)).astype(int)
                    if len(labeled_set)>len(subset):
                        l_mixed_set = len(subset)
                    else:
                        l_mixed_set = len(labeled_set) 
                    for cnt in range(2*int(l_mixed_set/BATCH)):
                        idx = int(cnt / 2)
                        if cnt % 2 == 0:
                            interleaved[cnt*BATCH:(cnt+1)*BATCH] = labeled_set[idx*BATCH:(idx+1)*BATCH]                             
                        else:
                            interleaved[cnt*BATCH:(cnt+1)*BATCH] = subset[idx*BATCH:(idx+1)*BATCH] 
                    interleaved = interleaved.tolist()



                    if args.dataset=='rafd':
                        if len(labeled_set)>len(subset):
                            interleaved = interleaved + labeled_set[(idx+1)*BATCH:]
                        else:
                            interleaved = interleaved + subset[(idx+1)*BATCH:]
                    
                        concat_dataset = ConcatDataset((data_train, data_unlabeled))
                        unlabelled_loader = DataLoader(concat_dataset, batch_size=BATCH, 
                                                        sampler=SubsetRandomSampler(interleaved), 
                                                        pin_memory=True, drop_last=drop_flag)
                        unlab_loader = DataLoader(data_train, batch_size=BATCH, 
                                                        sampler=SubsetRandomSampler(unlabeled_set), 
                                                        pin_memory=True, drop_last=drop_flag)
                    else:
                        # concat_dataset = ConcatDataset((data_train, data_unlabeled)) 
                        # interleaved_ext = [x + 50000 for x in labeled_set]
                        interleaved = interleaved #+ interleaved_ext[:2000]
                        unlabelled_loader = DataLoader(data_train, batch_size=BATCH, 
                                                        sampler=SubsetRandomSampler(interleaved), 
                                                        pin_memory=True, drop_last=drop_flag)
                        unlab_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                                        sampler=SubsetRandomSampler(unlabeled_set), 
                                                        pin_memory=True, drop_last=drop_flag)
                if (args.method_type == "JLS") or (args.method_type == "TJLS"):
                    dataloaders  = {'train': train_loader, 'multi': unlabelled_loader, 
                                    'test': test_loader, 'unlabeled': unlab_loader}
            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                if args.dataset == "fashionmnist":
                    if args.visual_transformer:
                        model    = resnet.ViTResNet18fm(num_classes=NO_CLASSES).cuda()
                    else:
                        model    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
                elif args.dataset == "rafd":
                    #resnet18    = vgg11().cuda() 
                    if args.visual_transformer:
                        if (args.method_type == "JLS"):
                            model = resnet.ResNet18(NO_CLASSES, avg_pool=16).cuda()
                        else:
                            model = resnet.ViTResNet18(NO_CLASSES, avg_pool=16).cuda()
                    else:
                        model    = resnet.ResNet18(num_classes=NO_CLASSES, avg_pool=16).cuda()
                else:
                    if args.visual_transformer:
                        if (args.method_type == "JLS"):
                            model    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
                        else:
                        # model    = resnet.VTWide_ResNet28(num_classes=NO_CLASSES).cuda()
                            model    = resnet.ViTResNet18(num_classes=NO_CLASSES).cuda()
                        # model    = resnet.VTvgg16(NO_CLASSES).cuda()
                    else:
                        model    = resnet.ResNet18(num_classes=NO_CLASSES, avg_pool=4).cuda()
                        # model    = resnet.vgg16(NO_CLASSES).cuda()

                    no_param = 0
                    for parameter in model.parameters():
                        a = parameter.reshape(-1).size()
                        no_param += a[0]
                    print(no_param)
                if method == 'lloss':
                    #loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
                    loss_module = LossNet().cuda()

            models      = {'backbone': model}
            if method =='lloss':
                models = {'backbone': model, 'module': loss_module}
            
            if (args.method_type == "JLS") or (args.method_type == "TJLS"):
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    sampler_module = Discriminator2(512).cuda()
                    models = {'backbone': model, 'sampler': sampler_module}
                no_param = 0
                for parameter in sampler_module.parameters():
                    a = parameter.reshape(-1).size()
                    no_param += a[0]
                print(no_param)
            torch.backends.cudnn.benchmark = True

            # MILESTONES = [90,120,150,180]
            # WDECAY = 4e-5
            # MOMENTUM = 0.9     
            Nes_flag = True        
            # Loss, criterion and scheduler (re)initialization

            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                momentum=MOMENTUM, weight_decay=WDECAY, nesterov=Nes_flag)
                # optim_backbone = optim.Adam(models['backbone'].parameters(), lr=0.001)
 
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss':
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

            if (args.method_type == "JLS") or (args.method_type == "TJLS"):
                optim_sampler   = optim.SGD(models['sampler'].parameters(), lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY, nesterov=Nes_flag)
                # optim_sampler   = optim.Adam(models['sampler'].parameters(), lr=LR_GCN, 
                #                             weight_decay=WDECAY)
                sched_sampler   = lr_scheduler.MultiStepLR(optim_sampler, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'sampler': optim_sampler}
                schedulers = {'backbone': sched_backbone, 'sampler': sched_sampler}
            # Training and testing
            if (args.method_type == "JLS") or (args.method_type == "TJLS"):
                # sets = [labeled_set, unlabeled_set]
                arg = train_with_sampler(models, method, criterion, optimizers, schedulers, dataloaders, 
                                args.no_of_epochs, EPOCHL, args, len(labeled_set), len(subset[:5000+cycle*2500]))
                acc = test_with_sampler(models, EPOCH, method, dataloaders, args, mode='test')
            else:
                train(models, method, criterion, optimizers, schedulers, dataloaders, 
                      args.no_of_epochs, EPOCHL, args, unlabeled_set, labeled_set, data_unlabeled)
                acc = test(models, EPOCH, method, dataloaders, args, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")


            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            if (args.method_type != "JLS") and (args.method_type != "TJLS"):
                arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args, synth_set, drop_flag, NUM_TRAIN)

            # Update the labeled dataset and the unlabeled dataset, respectively

            if args.dataset == "rafd":
                synth_set += list(torch.tensor(indices)[arg][-ADDENDUM:].numpy())
                shifted_synth_set = [ x + NUM_TRAIN for x in synth_set]
                concat_dataset = ConcatDataset((data_train, data_unlabeled))
                concat_set = list(np.arange(0, NUM_TRAIN,1)) + shifted_synth_set
                dataloaders['train'] = DataLoader(concat_dataset, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(concat_set), 
                                            pin_memory=True, drop_last=drop_flag)
                listd = list(torch.tensor(unlabeled_set)[arg][:-ADDENDUM].numpy()) 
                unlabeled_set = listd #+ unlabeled_set[SUBSET:]
            else:
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True, drop_last=drop_flag)
                listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
                unlabeled_set = listd #+ unlabeled_set[SUBSET:]
            if args.dataset == "rafd":
                print(len(shifted_synth_set), min(shifted_synth_set), max(shifted_synth_set))
            else:
                print(len(labeled_set), min(labeled_set), max(labeled_set))

    results.close()
