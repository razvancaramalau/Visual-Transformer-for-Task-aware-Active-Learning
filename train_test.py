from PIL import Image
from config import *
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import models.resnet as resnet
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def test(models, epoch, method, dataloaders, args, mode='val'):
    # assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total

    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                scores, _, _ = models['backbone'](inputs)
                # output = F.log_softmax(scores, dim=1)
                # loss =  F.nll_loss(output, labels, reduction="sum")
                _, preds = torch.max(scores.data, 1)
                # total_loss += loss.item()
                # _, preds = torch.max(output, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                # correct += preds.eq(labels).sum()
        
        return 100 * correct / total

def test_with_sampler(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'vtgcn':
        models['sampler'].eval()
    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total


iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):


    models['backbone'].train()
    if method == 'lloss':
        models['module'].train()
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            
            inputs = data[0].cuda()
            labels = data[1].cuda()


        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs) 
        target_loss = criterion(scores, labels)
        # target_loss =  F.nll_loss(F.log_softmax(scores, dim=1), labels)
        if method == 'lloss':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss
        # loss = target_loss
        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss':
            optimizers['module'].step()
    return loss


def train_epoch_with_sampler(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, l_lab, l_ulab):


    
    # global iters
    sampler_criterion = nn.BCELoss()
    # disc_lbl = torch.cat((torch.ones((cycle+1)*ADDENDUM), torch.zeros(SUBSET)),0).cuda()
    
    if l_ulab>=l_lab:
        disc_lbl = torch.zeros((BATCH*len(dataloaders['multi'])))
        for cnt in range(0, 2*int(l_lab/BATCH), 2):
            disc_lbl[cnt*BATCH:(cnt+1)*BATCH] = torch.ones((BATCH))
    else:
        disc_lbl = torch.ones((BATCH*len(dataloaders['multi'])))
        for cnt in range(1, 2*int(l_ulab/BATCH), 2):
            disc_lbl[cnt*BATCH:(cnt+1)*BATCH] = torch.zeros((BATCH))
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        disc_lbl = disc_lbl.cuda()
    iters = 0
    models['backbone'].train()
    models['sampler'].train()
    for data in tqdm(dataloaders['multi'], leave=False, total=len(dataloaders['multi'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].cuda()

        sampler_labels = disc_lbl[iters*BATCH:(iters+1)*BATCH]

        iters += 1
        
        
        optimizers['backbone'].zero_grad()
        optimizers['sampler'].zero_grad()
        scores, features, _ = models['backbone'](inputs) 
       
        # Get features 
        features = nn.functional.normalize(features)
        # Train sampler
        outputs = models['sampler'](features)
        # print(max(outputs),min(outputs))
        loss_sampler = sampler_criterion(outputs, torch.unsqueeze(sampler_labels, 1))

        if torch.sum(sampler_labels)==BATCH:
            target_loss = criterion(scores, labels)
            # target_loss =  F.nll_loss(F.log_softmax(scores, dim=1), labels)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0) 
            loss = m_backbone_loss + 0.5 *loss_sampler
        else:
            loss =  0.5 *loss_sampler

        #print(loss)
        loss.backward()
        optimizers['backbone'].step()
        optimizers['sampler'].step()

        

    return loss
    
def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, args, subset, labeled_set, data_unlabeled):
    
    print('>> Train a Model.')
    

    best_acc = 0.
    
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)
        # print(loss)
        schedulers['backbone'].step()
        if method == 'lloss':
            schedulers['module'].step()

        if True and epoch % 20  == 1:
            # print(loss)
            acc = test(models, epoch, method, dataloaders, args, mode='test')
            # acc = test(models, dataloaders, mc, 'test')
            if args.dataset == 'icvl':

                if best_acc > acc:
                    best_acc = acc
                print('Val Error: {:.3f} \t Best Error: {:.3f}'.format(acc, best_acc))
            else:
                if best_acc < acc:
                    best_acc = acc
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')


def train_with_sampler(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, 
                       epoch_loss, args, l_lab, l_ulab):
    print('>> Train a Model.')
    best_acc = 0.
    
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch_with_sampler(models, method, criterion, optimizers, dataloaders, 
                                        epoch, epoch_loss, l_lab, l_ulab)

        schedulers['backbone'].step()
        if (method == "JLS") or (method == "TJLS"):
            schedulers['sampler'].step()

        if True and epoch % 20  == 1:
            acc = test_with_sampler(models, epoch, method, dataloaders, args, mode='test')

            # acc = test(models, dataloaders, mc, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

    print('>> Finished.')
    models['sampler'].eval()
    models['backbone'].eval()    

    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        scores = torch.tensor([]).cuda() 

    with torch.no_grad():
        for inputs, labels, _ in dataloaders['unlabeled']:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()
            _, features, _ = models['backbone'](inputs)
            scores_batch = models['sampler'](features)
            scores = torch.cat((scores, scores_batch), 0)
      

    s_margin = 0.1
    scores = torch.squeeze(scores.detach().cpu())
    scores_median = np.abs(scores.numpy() - s_margin)

    arg = np.argsort(-(scores_median))

    
    return arg

def train_epoch_exp1(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, random_subset):


    models['backbone'].train()
    if method == 'lloss':
        models['module'].train()
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            # print(max(data[2]))
            for i in random_subset:
                data[1][data[2]==i] = 10
            # print(max(data[1]))
            labels = data[1].cuda()
            # labels[random_subset] = 10


        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs) 
        target_loss = criterion(scores, labels)
        # target_loss =  F.nll_loss(F.log_softmax(scores, dim=1), labels)
        if method == 'lloss':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss
        # loss = target_loss
        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss':
            optimizers['module'].step()
    return loss

def train_exp1(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, args, random_subset):
    print('>> Train a Model.')
    best_acc = 0.
    
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch_exp1(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, random_subset)

        schedulers['backbone'].step()
        if method == 'lloss':
            schedulers['module'].step()

        if False and epoch % 40  == 1:
            acc = test(models, epoch, method, dataloaders, args, mode='test')
            # acc = test(models, dataloaders, mc, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
