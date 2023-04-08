import dataset
from models import CLIPFormer
import torch
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from utils import *
from sklearn import metrics
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
n_layers = 6
num_bags = 32
num_features = 7200 // num_bags
model = CLIPFormer(num_layers=n_layers, emb_size=512).to(device=device)

#hyperparams
batch_size = 32
#num_bags = 32
#num_features = 7200 // num_bags
lr = 5e-5
wd = 0.001
optim = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
loss_fn = MIL

print(f'Device {device}\nbatch_size {batch_size}\nLr {lr}\n Wd {wd}')
#dataset and dataloader
abnormal_feat_dir = r'/local/scratch/c_adabouei/video_analysis/dataset/CLIP_Interpolated/Abnormal_Interpolated_Features_950'
normal_feat_dir = r"/local/scratch/c_adabouei/video_analysis/dataset/CLIP_Interpolated/Normal_Interpolated_Features_520"
train_split = r"/local/scratch/c_adabouei/video_analysis/dataset/UCF_Crime_Complete/Anomaly_Train.txt"
test_list_combined = r"/local/scratch/c_adabouei/video_analysis/dataset/UCF_Crime_Complete/test_complete.txt"
test_feat_path = r"/local/scratch/c_adabouei/video_analysis/dataset/CLIP_Interpolated/Test_Videos_Interpolated"
save_dir =  r"/local/scratch/c_adabouei/video_analysis/dataset/UCF_Crime_Complete/save_dir"

# Dataset
train_ds = dataset.Dataset_train(abnormal_feat_dir, normal_feat_dir)
valid_ds = dataset.Dataset_valid(test_feat_path)

# Dataloaders
train_dl = DataLoader(train_ds, batch_size=batch_size, 
                                      shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size=1, 
                                      shuffle=True, num_workers=8, pin_memory=False, drop_last=False)

def seed_everything(seed: int = 10):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(epoch):
    print(f'Epoch number {epoch+1}')
    model.train()
    running_loss = 0.0
    print('Training begins')
    for i, (abnormal_inputs, normal_inputs) in enumerate(tqdm(train_dl)):
        s_time = time.time()
        
        abnormal_inputs = abnormal_inputs.to(device)
        normal_inputs = normal_inputs.to(device)
                        # print("normal_inputs_shape", normal_inputs.shape) # [batch_size, num_bags, 1]
        abnormal_scores = model(abnormal_inputs) 
        normal_scores = model(normal_inputs)  
        combined_anomaly_scores = torch.cat([abnormal_scores, normal_scores], dim=0)
        combined_anomaly_scores = torch.sigmoid(combined_anomaly_scores)
                                                            
        loss = MIL(combined_anomaly_scores, batch_size)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss += loss.item()
        e_time = time.time()
    print('train loss = {}'.format(running_loss/len(train_dl)))


def test(epoch):
    model.eval()
    auc = 0
    all_gts = []
    all_scores = []

    with torch.no_grad():
        for valid_segments, video_num_frames, video_gt in tqdm(valid_dl):
            valid_segments =  valid_segments.to(device)
            video_num_frames = video_num_frames.to(device)

            anomaly_scores = model(valid_segments.squeeze(0))
            anomaly_scores = anomaly_scores.squeeze(0).squeeze(1)
            anomaly_scores = anomaly_scores.cpu().detach().numpy()

            # making the predictions compatible with gt list
            if video_num_frames > num_features:
                anomaly_scores = [anomaly_score for anomaly_score in anomaly_scores for i in range(num_features)]
                all_scores.extend(anomaly_scores)
            anomaly_score_last = anomaly_scores[-1]
            remainder = video_num_frames % num_features

            if remainder != 0:
                all_scores.extend(anomaly_score_last for i in range(remainder))  

            video_gt_list = torch.zeros(video_num_frames)

            if video_gt == 1:
                all_gts.extend(video_gt_list)
            else:
                video_gt_list[video_gt[0]:video_gt[1]] = 1
                try:
                    video_gt_list[video_gt[2]:video_gt[3]] = 1
                except:
                    pass
                all_gts.extend(video_gt_list)
    auc = metrics.roc_auc_score(all_gts, all_scores)
    fpr, tpr, th = metrics.roc_curve(all_gts, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc


best_auc = 0.5
auc_scores = []

epochs = 20
seed_everything(0)
for epoch in range(epochs):
    train(epoch)
    print('Starting validation')
    epoch_auc = test(epoch)
    auc_scores.append(epoch_auc)

    if epoch_auc > best_auc:
        save_path = "./checkpoints/model_tf_best.pth"
        torch.save(model.state_dict(), save_path)
        best_auc = epoch_auc

    print(f'AUC score per epoch {epoch_auc}')
    print(f'Best AUC score {best_auc}')

plt.figure()
plt.plot(list(range(1, epochs+1)), auc_scores)
plt.xlabel("epochs")
plt.ylabel("AUC score")
plt.savefig("./plots/AUC_bs_32_epoch_20_L6_no_embed_updated_fcnn.png")



    


