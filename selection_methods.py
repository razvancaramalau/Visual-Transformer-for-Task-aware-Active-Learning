import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torch.distributions import Bernoulli
# Custom
from rewards import compute_reward
# from main_vt import NO_CLASSES
from config import *
from models.query_models import VAE, Discriminator, GCN, DSN
from data.sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    
    vae = models['vae']
    discriminator = models['discriminator']
    vae.train()
    discriminator.train()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
    
    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 2

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int( (ADDENDUM*cycle+ SUBSET) * EPOCHV / BATCH )
    train_iterations = 901
    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)
            
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
            if iter_count % 100 == 0:
                print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
#
def entropy(p, dim = -1, keepdim = None):
   return torch.sum(-torch.where(p > 0, p * p.log(), p.new([0.0])), dim=dim) # can be a scalar, when PyTorch.supports it

def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = entropy(pred_loss)
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()    
    with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                    _, features_batch, _ = models['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features #.detach().cpu().numpy()
    return feat

def get_kcg(models, labeled_data_size, unlabeled_loader, subset):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
        labels_batch = torch.tensor([], dtype=torch.long).cuda() 

    with torch.no_grad():
        for inputs, labels, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
            labels_batch = torch.cat((labels_batch, labels), 0)
        feat = features.detach().cpu().numpy()
        # feat = features_batch.detach().cpu().numpy()
        # np.save("feat_s.npy", feat)
        # np.save("labels_s.npy", labels_batch.detach().cpu().numpy())

        new_av_idx = np.arange(subset,(subset + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, 2500)
        # print(min(batch), max(batch))
        other_idx = [x for x in range(subset) if x not in batch]
    # np.save("selected_s.npy", batch)
    return  other_idx + batch

def get_kcg2(models, unlabeled_data_size, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(unlabeled_data_size,(unlabeled_data_size + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(unlabeled_data_size) if x not in batch]
    return  other_idx + batch

# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args, synth_set, drop_flag, NUM_TRAIN):
    SUBSET = NUM_TRAIN
    if method == 'Random':
        arg = np.random.randint(len(subset), size=len(subset))
    
    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True, drop_last=drop_flag)
        if drop_flag:
            if cycle==0:
                limit_subset = 5000
            else:
                limit_subset = 5000 + int((SUBSET+(cycle)*ADDENDUM)/BATCH) * BATCH - SUBSET
        else:
            if cycle==0:
                limit_subset = 5000
            else:
                limit_subset = 5000 + ADDENDUM*(cycle)
        arg = get_kcg(model, limit_subset, unlabeled_loader, 49920-limit_subset)
        # print(min(arg), max(arg))
    if method == 'lloss':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True, drop_last=drop_flag)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)        

    if method == 'VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True, drop_last=drop_flag)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True, drop_last=drop_flag)
        if args.dataset == 'fashionmnist':
            vae = VAE(28,1,3)
            discriminator = Discriminator(28)
        else:
            vae = VAE()
            discriminator = Discriminator(32)
        models      = {'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1)
        
        all_preds, all_indices = [], []
        #         break
        for images, _, indices in unlabeled_loader:                       
            images = images.cuda()
            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds) 


    if method == 'CDAL':
        if args.dataset == 'cifar100':
            hidd_dim = 1024
            number_of_picks = 2000
        else:
            hidd_dim = 256
            number_of_picks = 100
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset[:10000]), # more convenient if we maintain the order of subset
                                            pin_memory=True, drop_last=drop_flag)
        # features = get_features(model, unlabeled_loader)
        all_preds = []
        for images, _, indices in unlabeled_loader:                       
            images = images.cuda()
            with torch.no_grad():
                preds, _, _ = model['backbone'](images)
                preds = preds.cpu().data
                all_preds.extend(preds)
        
        # features = torch.from_numpy(np.load("CDAL/features/000017.npy"))
        features = torch.Tensor(torch.stack(all_preds)).cuda()
        features = nn.functional.normalize(features, dim=1)
        No_classes = features.shape[1]
        model = DSN(in_dim=No_classes, hid_dim=hidd_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-02, weight_decay=1e-05)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        start_epoch = 0
        max_epoch = 20
        # start = 0
        model.train()
        baseline=0.
        best_reward=0.0
        best_pi=[]
        use_gpu = True
        num_episode = 1

        beta = 0.01
        arg = []
        if use_gpu: model = model.cuda()
        for epoch in range(start_epoch, max_epoch):
            seq = features
            seq = seq.unsqueeze(0) # input shape (1, seq_len, dim)
            if use_gpu: seq = seq.cuda()
            probs = model(seq) # output shape (1, seq_len, 1)
            cost = beta * (probs.mean() - 0.5)**2 
            m = Bernoulli(probs)
            epis_rewards = []
            for _ in range(num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                reward,pick_idxs = compute_reward(seq, actions, probs, nc=No_classes, picks=number_of_picks, use_gpu=use_gpu)
                if(reward>best_reward):
                    best_reward=reward
                    best_pi=pick_idxs
                expected_reward = log_probs.mean() * (reward - baseline)
                cost -= expected_reward # minimize negative expected reward
                epis_rewards.append(reward.item())

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            print("epoch {}/{}\t reward {}\t".format(epoch+1, max_epoch, np.mean(epis_rewards)))
            # f=open('selection/'+str(start)+'.txt','w')

        other_idx = [x for x in range(SUBSET - (cycle+1) * 100) if x not in best_pi]
        arg = other_idx + best_pi.tolist()
            # f.close()
    return arg
