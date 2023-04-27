Config = dict(
batch_size = 1,
epochs = 5, # Number of epochs to train after generating and adding pseudo-labels once
ssl_steps= 30, # Number of times to generate pseudo-labels
percent_select = 0.5,
optimizer = "Adam",
lr = 1e-5,
n_heads = 4, # number of Transformer heads
num_layers = 1, # number of Transformer layers
weight_decay = 0.001,
seed = 1,
device = "cuda:7",
use_wandb = False,
lr_scheduler = None,
interpolation_dimension = 64,
wandb_project_name = "UCF_Crime_SSL",
notes = "max score, hacs")

print([(key, value) for key, value in Config.items()])

abnormal_feat_dir = r'/local/scratch/v_eshaan_mandal/Anomaly'
normal_feat_dir = r"/local/scratch/v_eshaan_mandal/Normal"
unlabelled_feat_dir = r"/local/scratch/c_adabouei/video_analysis/dataset/HACS_Features"
train_split = r"anomaly_normal.txt"
test_list_combined = r"test_complete.txt"
save_dir =  r"/local/scratch/c_adabouei/video_analysis/dataset/HACS_Exp/save_dir"

import os
import gc
from tqdm import tqdm
import random
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchmetrics
from itertools import cycle

def wandb_log(**kwargs):
        """
        Logs a key-value pair to W&B
        """
        for k, v in kwargs.items():
            wandb.log({k: v})


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def MIL(combined_anomaly_scores, batch_size):
    combined_anomaly_scores = combined_anomaly_scores.squeeze(2) 
    loss = 0 
    for i in range(batch_size):
        y_abnormal_max = torch.max(combined_anomaly_scores[i, :])
        y_normal_max = torch.max(combined_anomaly_scores[i + batch_size, :]) 
        loss = (F.relu(1. - y_abnormal_max + y_normal_max)) + loss 
    return loss/batch_size


def BCE_Loss(anomaly_scores, ground_truth):
    anomaly_scores = anomaly_scores.squeeze()
    if ground_truth == 0:
        loss = F.binary_cross_entropy(anomaly_scores, torch.zeros_like(anomaly_scores))    
    else:
        # anomaly_scores.sort()
        abnormal_segments_score = [anomaly_score for anomaly_score in anomaly_scores if anomaly_score >= 0.7]
        abnormal_segments_score = torch.tensor(abnormal_segments_score)
        abnormal_loss = F.binary_cross_entropy(abnormal_segments_score, torch.ones_like(abnormal_segments_score))
        normal_segments_score = [anomaly_score for anomaly_score in anomaly_scores if anomaly_score <= 0.5]
        normal_segments_score = torch.tensor(normal_segments_score)
        normal_loss = F.binary_cross_entropy(normal_segments_score, torch.zeros_like(normal_segments_score))
        w = normal_loss/abnormal_loss
        abnormal_loss = abnormal_loss*w
        # if (torch.isnan(abnormal_loss)).item():
        #     return normal_loss
        loss = abnormal_loss + normal_loss
        if torch.isnan(loss).item():
            loss = 0
    return loss


def unlabelled_predict(model, loader, epoch, device):
    model.eval()
    anomaly_score_dict = {}
    with torch.no_grad():
        for unlabelled_segments, index in tqdm(loader):  
            unlabelled_segments =  unlabelled_segments.to(device)
            anomaly_scores = model(unlabelled_segments)
            anomaly_scores = anomaly_scores.squeeze(0).squeeze(1)
            anomaly_scores = anomaly_scores.cpu().detach().numpy()
            # anomaly_score = sum(score for score in anomaly_scores)/len(anomaly_scores)
            anomaly_score = max(score for score in anomaly_scores)
            anomaly_score_dict[index] = anomaly_score
    return anomaly_score_dict
    

def unlabelled_select(anomaly_score_dict, percent_select):
    n_top = int(len(anomaly_score_dict) * percent_select/2)
    sorted_dict = sorted(anomaly_score_dict.items(), key=lambda x: x[1])
    abnormal_index = [k.item() for k, v in sorted_dict[-n_top:]]
    normal_index = [k.item() for k, v in sorted_dict[:n_top]]
    return abnormal_index, normal_index     


seed_everything(Config['seed'])

n_heads = Config['n_heads']
num_layers = Config['num_layers']
batch_size = Config['batch_size']
optimizer = Config['optimizer']
lr = Config['lr']
epochs = Config['epochs']
ssl_steps = Config["ssl_steps"]
percent_select = Config['percent_select']
weight_decay = Config['weight_decay']
use_wandb = Config['use_wandb']
device = Config['device']
interpolation_dimension = Config['interpolation_dimension']

device = torch.device(device if torch.cuda.is_available() else "cpu")
print(device)


if Config['use_wandb'] == True:
    import wandb    
    wandb.login()    
    run = wandb.init(project=Config["wandb_project_name"], entity='video_anomaly_detection', config=Config,
        job_type='train',anonymous='allow')



# Train Dataset
class Dataset_(Dataset):    
    def __init__(self, abnormal_feat_dir, normal_feat_dir, interpolation_dimension, train_split):   
        self.abnormal_path = abnormal_feat_dir 
        self.normal_path = normal_feat_dir  
        self.interpolation_dimension = interpolation_dimension 
        
        #  Creating splits using the txt file provided

        with open(train_split, 'r') as f:
            self.train_list = f.readlines()

        self.abnormal_train_list = [vid for vid in self.train_list if 'Normal' not in vid]
        # self.abnormal_train_list = [vid.split('/')[1][:-5] for vid in self.abnormal_train_list]   
        self.normal_train_list = [vid for vid in self.train_list if 'Normal' in vid]
        # self.normal_train_list = [vid.split('/')[1][:-5] for vid in self.normal_train_list]  
              
    def __len__(self):          
        return len(self.normal_train_list)

    def _create_segments(self, feature_dir, video_num):
        video_features = np.load(os.path.join(feature_dir, video_num))
        video_features = torch.from_numpy(video_features).to(torch.float32)
        video_features = video_features.reshape(-1, 512).unsqueeze(0) # [1, frames, 512]
        x = video_features.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.interpolation_dimension, 1), mode='bilinear')
        x = x.squeeze(-1).squeeze(0).permute(1, 0)   
        return x

    def __getitem__(self, index): 
        abnormal_video_num = self.abnormal_train_list[index]
        normal_video_num = self.normal_train_list[index]    
        abnormal_segments = self._create_segments(self.abnormal_path, abnormal_video_num[:-1])        
        normal_segments = self._create_segments(self.normal_path, normal_video_num[:-1])       
        # print("normal_segmemts_shape", normal_segments.shape) #[bags, 512]
        return abnormal_segments, normal_segments

        
# Valid Dataset
class Dataset_valid(Dataset):    
    def __init__(self, abnormal_feat_dir, normal_feat_dir, interpolation_dimension, test_list_combined):   
        self.abnormal_path = abnormal_feat_dir 
        self.normal_path = normal_feat_dir
        self.interpolation_dimension = interpolation_dimension 

        with open(test_list_combined, 'r') as f:
            self.test_list = f.readlines()
        
    def __len__(self):        
        return len(self.test_list)  
        
    def _create_segments(self, clip_feat):
        video_features = torch.from_numpy(clip_feat).to(torch.float32)
        video_features = video_features.reshape(-1, 512).unsqueeze(0)
        x = video_features.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.interpolation_dimension, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)   
        return x.squeeze(0)       

    def __getitem__(self, idx):
        if 'Normal' in self.test_list[idx]:
            name, num_frames, extra = self.test_list[idx].split(' ')
            video_gts = int(extra[1])
            clip_feat = np.load(os.path.join(self.normal_path, name))
        else:
            name, num_frames, extra = self.test_list[idx].split('|')
            video_gts = extra[1:-2].split(',')
            video_gts = [int(i) for i in video_gts]
            clip_feat = np.load(os.path.join(self.abnormal_path, name))

        valid_segments = self._create_segments(clip_feat)
        return valid_segments, int(num_frames), video_gts


# Dataset for unlabelled videos
class Dataset_unlabelled_1(Dataset):    
    def __init__(self, unlabelled_feat_dir, interpolation_dimension, unlabelled_select_list=None):   
        self.unlabelled_path = unlabelled_feat_dir
        self.interpolation_dimension = interpolation_dimension 
        self.unlabelled_select_list = unlabelled_select_list
        if self.unlabelled_select_list is None:
            self.unlabelled_list = [vid for vid in os.listdir(self.unlabelled_path) if "clip.npy" in vid]
        else:
            self.unlabelled_list = self.unlabelled_select_list

    def __len__(self):        
        return len(self.unlabelled_list)  
        
    def _create_segments(self, clip_feat):
        video_features = torch.from_numpy(clip_feat).to(torch.float32)
        video_features = video_features.reshape(-1, 512).unsqueeze(0)
        x = video_features.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.interpolation_dimension, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)   
        return x.squeeze(0)       

    def __getitem__(self, index):      
        unlabelled_video_num = self.unlabelled_list[index]
        clip_feat = np.load(os.path.join(self.unlabelled_path, unlabelled_video_num))
        unlabelled_segments = self._create_segments(clip_feat)     
        return unlabelled_segments, index


class Dataset_unlabelled(Dataset):    
    def __init__(self, unlabelled_feat_dir,
     abnormal_unlabelled_index, normal_unlabelled_index, interpolation_dimension, train_split):   
        # self.abnormal_path = abnormal_feat_dir 
        # self.normal_path = normal_feat_dir  
        self.unlabelled_path = unlabelled_feat_dir
        self.abnormal_unlabelled_index_list = abnormal_unlabelled_index
        self.normal_unlabelled_index_list = normal_unlabelled_index
        self.interpolation_dimension = interpolation_dimension        

        with open(train_split, 'r') as f:
            self.train_list = f.readlines()

        # self.abnormal_UCF_path_list = [os.path.join(self.abnormal_path, vid)[:-1] for vid in self.train_list if 'Normal' not in vid]
        # self.normal_UCF_path_list = [os.path.join(self.normal_path, vid)[:-1] for vid in self.train_list if 'Normal' in vid]
        # self.abnormal_UCF_path_dict = {item: 'L' for item in self.abnormal_UCF_path_list}
        # self.normal_UCF_path_dict = {item: 'L' for item in self.normal_UCF_path_list}

        self.abnormal_pseudolabel_path_list = [os.path.join(self.unlabelled_path, os.listdir(self.unlabelled_path)[i]) for i in self.abnormal_unlabelled_index_list]
        self.normal_pseudolabel_path_list = [os.path.join(self.unlabelled_path, os.listdir(self.unlabelled_path)[i]) for i in self.normal_unlabelled_index_list]
        # self.abnormal_pseudolabel_path_dict = {item: 'U' for item in self.abnormal_pseudolabel_path_list}
        # self.normal_pseudolabel_path_dict = {item: 'U' for item in self.normal_pseudolabel_path_list}

        # self.concatenated_abnormal_path_dict = self.abnormal_UCF_path_dict | self.abnormal_pseudolabel_path_dict
        # self.concatenated_normal_path_dict = self.normal_UCF_path_dict | self.normal_pseudolabel_path_dict

        # self.concatenated_abnormal_path_list = list(self.concatenated_abnormal_path_dict.keys())
        # self.concatenated_normal_path_list = list(self.concatenated_normal_path_dict.keys())
        # self.concatenated_abnormal_path_value_list = list(self.concatenated_abnormal_path_dict.values())
        # self.concatenated_normal_path_value_list = list(self.concatenated_normal_path_dict.values())


    def __len__(self):          
        return len(self.normal_pseudolabel_path_list) 

    def _create_segments(self, video_num):
        video_features = np.load(video_num)
        video_features = torch.from_numpy(video_features).to(torch.float32)
        video_features = video_features.reshape(-1, 512).unsqueeze(0) # [1, frames, 512]
        x = video_features.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.interpolation_dimension, 1), mode='bilinear')
        x = x.squeeze(-1).squeeze(0).permute(1, 0)   
        return x

    def __getitem__(self, index): 

        abnormal_video_path = self.abnormal_pseudolabel_path_list[index]
        normal_video_path = self.normal_pseudolabel_path_list[index]  
        # abnormal_value = self.concatenated_abnormal_path_value_list[index]
        # normal_value = self.concatenated_normal_path_value_list[index]
        abnormal_segments = self._create_segments(abnormal_video_path)     
        normal_segments = self._create_segments(normal_video_path)   
        # abnormal_dict = {k:v for k, v in zip(abnormal_segments, abnormal_value)}
        # normal_dict = {k:v for k, v in zip(normal_segments, normal_value)}

        return abnormal_segments, normal_segments



import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, n_heads, num_layers):
        super(Transformer, self).__init__()        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)     
        self.apply(self.init_weight)

    def forward(self, x):
        out= self.transformer_encoder(x)
        return out
    
    def init_weight(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
                torch.nn.init.normal_(child.weight, mean=0.2, std=0.7)
            elif isinstance(child, nn.BatchNorm2d):
                child.weight.data.fill_(1)
                child.bias.data.zero_()
            else:
                self.init_weight(child)
    
   
class FCNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.apply(self.init_weight)
    
    def init_weight(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
                torch.nn.init.normal_(child.weight, mean=0.2, std=0.7)
            elif isinstance(child, nn.BatchNorm2d):
                child.weight.data.fill_(1)
                child.bias.data.zero_()
            else:
                self.init_weight(child)
   
    
class Model(nn.Module):    
    def __init__(self, n_heads, num_layers, device):  
        super(Model, self).__init__()
        self.Transformer_Encoder = Transformer(n_heads, num_layers)
        self.FCNN = FCNN()  
        self.to(device)  
        
    def forward(self, x):
        x = self.Transformer_Encoder(x)
        x = self.FCNN(x)
        return x
    

model = Model(n_heads, num_layers, device)
if Config['use_wandb'] == True:
    wandb.watch(model)

model = model.to(device)
params = list(model.parameters()) 

if optimizer == "Adam":
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)  
elif optimizer == "SGD":
    optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)    
elif optimizer == "Adagrad":
    optimizer = torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay) 

def train_once(ssl_step, epoch, model, loader, batch_size, optimizer, device):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    print('Training...')

    for (abnormal_inputs, normal_inputs) in tqdm(loader):
        abnormal_inputs = abnormal_inputs.to(device)
        normal_inputs = normal_inputs.to(device)
        # print("normal_inputs_shape", normal_inputs.shape) # [batch_size, num_bags, 1]
        abnormal_scores = model(abnormal_inputs) 
        normal_scores = model(normal_inputs)  
        combined_anomaly_scores = torch.cat([abnormal_scores, normal_scores], dim=0)
        # print("combined_anomaly_scores.size", combined_anomaly_scores.shape) # [2*batch_size, num_bags, 1]
        loss = MIL(combined_anomaly_scores, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('train loss = {}'.format(train_loss/len(loader)))


def train(ssl_step, epoch, model, labelled_loader, unlabelled_loader, batch_size, optimizer, device):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    print('Training...')

    # if ssl_step == 0:
    #     for (abnormal_inputs, normal_inputs) in tqdm(loader):
    #         abnormal_inputs = abnormal_inputs.to(device)
    #         normal_inputs = normal_inputs.to(device)
    #         # print("normal_inputs_shape", normal_inputs.shape) # [batch_size, num_bags, 1]
    #         abnormal_scores = model(abnormal_inputs) 
    #         normal_scores = model(normal_inputs)  
    #         combined_anomaly_scores = torch.cat([abnormal_scores, normal_scores], dim=0)
    #         # print("combined_anomaly_scores.size", combined_anomaly_scores.shape) # [2*batch_size, num_bags, 1]
    #         loss = MIL(combined_anomaly_scores, batch_size)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()


    # else:    
    for labelled_batch, unlabelled_batch in tqdm(zip(labelled_loader, cycle(unlabelled_loader)), total=max(len(unlabelled_loader), len(labelled_loader))):

        labelled_abnormal, labelled_normal = labelled_batch
        unlabelled_abnormal, unlabelled_normal = unlabelled_batch

        labelled_abnormal = labelled_abnormal.to(device)
        labelled_normal = labelled_normal.to(device)

        labelled_abnormal_scores = model(labelled_abnormal)
        labelled_normal_scores = model(labelled_normal)
        
        unlabelled_abnormal = unlabelled_abnormal.to(device)
        unlabelled_normal = unlabelled_normal.to(device)

        unlabelled_abnormal_scores = model(unlabelled_abnormal)
        unlabelled_normal_scores = model(unlabelled_normal)

        combined_abnormal_scores = torch.cat([labelled_abnormal_scores, unlabelled_abnormal_scores], dim=0)
        combined_normal_scores = torch.cat([labelled_normal_scores, unlabelled_normal_scores], dim=0)

        combined_anomaly_scores = torch.cat([combined_abnormal_scores, combined_normal_scores], dim=0)
        loss = MIL(combined_anomaly_scores, batch_size) 


        # abnormal_dict = {k:v for k, v in zip(abnormal_scores, abnormal_value)}
        # normal_dict = {k:v for k, v in zip(normal_scores, normal_value)}
        # abnormal_inputs_unlabelled = [key for key, value in abnormal_dict.items() if value == 'U']
        
        # loss += loss + sum(BCE_Loss(anomaly_scores, 1) for anomaly_scores in abnormal_inputs_unlabelled)
        loss += loss + BCE_Loss(unlabelled_abnormal_scores, 1)

        # normal_inputs_unlabelled = [key for key, value in normal_dict.items() if value == 'U']
        # loss += loss + sum(BCE_Loss(anomaly_scores, 0) for anomaly_scores in normal_inputs_unlabelled)
        loss += loss + BCE_Loss(unlabelled_normal_scores, 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()           


    print('train loss = {}'.format(train_loss/(len(labelled_loader) + len(unlabelled_loader))))  


def test(model, loader, interpolation_dimension, device):
    print("Evaluating..")
    model.eval()
    auc = 0
    all_gt_list = []
    all_score_list = []

    with torch.no_grad():
        for valid_segments, video_num_frames, video_gt in tqdm(loader):  
            valid_segments, video_num_frames =  valid_segments.to(device), video_num_frames.to(device) 
            anomaly_scores = model(valid_segments)
            anomaly_scores = anomaly_scores.squeeze(0).squeeze(1)
            anomaly_scores = anomaly_scores.cpu().detach().numpy()

            # Replicate the score corresponding to 1 segment to all the frames present in that segment
            if video_num_frames > interpolation_dimension:
                repeat = video_num_frames // interpolation_dimension
                anomaly_scores = [anomaly_score for anomaly_score in anomaly_scores for i in range(repeat)]
                all_score_list.extend(anomaly_scores)
            anomaly_score_last = anomaly_scores[-1]
            remainder = video_num_frames % interpolation_dimension
            if remainder != 0:
                all_score_list.extend(anomaly_score_last for i in range(remainder))              

            # Creating the ground truth list using annotations provided 
            video_gt_list = torch.zeros(video_num_frames)

            # For normal videos, video_gt = 1
            if video_gt == 1:
                all_gt_list.extend(video_gt_list)
            else:
                video_gt_list[video_gt[0]:video_gt[1]] = 1
                try:
                    video_gt_list[video_gt[2]:video_gt[3]] = 1
                except:
                    pass
                all_gt_list.extend(video_gt_list)    

    all_gt_list = torch.tensor(all_gt_list)
    all_score_list = torch.tensor(all_score_list)
    auroc = torchmetrics.AUROC(task="binary")
    auc = auroc(all_score_list, all_gt_list) 
    return auc




train_dataset = Dataset_(abnormal_feat_dir, normal_feat_dir, interpolation_dimension, train_split)
labelled_train_batch = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=8)

valid_dataset = Dataset_valid(abnormal_feat_dir, normal_feat_dir, interpolation_dimension, 
                                test_list_combined)
labelled_valid_batch = DataLoader(valid_dataset, batch_size=1, 
                              shuffle=True, num_workers=8)

unlabelled_dataset_1 = Dataset_unlabelled_1(unlabelled_feat_dir, interpolation_dimension)
unlabelled_batch_1 = DataLoader(unlabelled_dataset_1, batch_size=1, 
                              shuffle=True, num_workers=8)



global best_auc
best_auc = 0.5
auc_scores = []


for ssl_step in range(0, ssl_steps):
    print("\n \n ---------------NEW_SSL_EPOCH_STARTS------------- SSL EPOCH NO -", ssl_step)
    for epoch in range(0, epochs):
        if ssl_step == 0:
            train_once(ssl_step, epoch, model, labelled_train_batch, batch_size, optimizer, device)
        else:
            train(ssl_step, epoch, model, labelled_train_batch, unlabelled_train_batch, batch_size, optimizer, device)
        epoch_auc = test(model, labelled_valid_batch, interpolation_dimension, device)
        auc_scores.append(epoch_auc)
        if epoch_auc > best_auc:
            model_save_path = os.path.join(save_dir, "transformer_fcnn_SSL_004.pth")
            torch.save(model.state_dict(), model_save_path)
            best_auc = epoch_auc
        print(f"The AUC SCORE is {epoch_auc}")
        print(f"The BEST AUC SCORE for the MIL Model is {best_auc}")
        if Config["use_wandb"] == True:
            wandb_log(AUC_score=epoch_auc)
    
    model.load_state_dict(torch.load(os.path.join(save_dir, "transformer_fcnn_SSL_004.pth")))
    model = model.to(device)
    
    print("Generating pseudo-labels")
    anomaly_score_dict = unlabelled_predict(model, unlabelled_batch_1, ssl_step, device)
    abnormal_unlabelled_index, normal_unlabelled_index = unlabelled_select(anomaly_score_dict, 
    percent_select)       

    
    unlabelled_dataset = Dataset_unlabelled(unlabelled_feat_dir, 
                            abnormal_unlabelled_index, normal_unlabelled_index, 
                            interpolation_dimension, train_split)
    unlabelled_train_batch = DataLoader(unlabelled_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=8)    
