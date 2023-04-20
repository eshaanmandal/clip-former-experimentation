import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torchvision.transforms import ToTensor
import torchvision as tv


class NormalVideo(Dataset):
    '''
        Custom dataset for Normal videos 
    '''
    def __init__(
            self,
            base_dir,
            frames = 3000, 
    ):
        self.base_dir = base_dir
        self.frames = frames
        
        with open('normal_train.txt', 'r') as f:
            self.list_of_videos = f.read().splitlines()


    def __len__(self):
        return len(self.list_of_videos)
    
    
    def interpolate(self, x, input_frames):
        # adding a batch
        x = x.unsqueeze(0)
        # [bs, frames, 512]

        x = x.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.frames, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)

        return x

        
    def __getitem__(self, idx: int):
        anomaly_clip = np.load(os.path.join(self.base_dir, self.list_of_videos[idx]))
        x = torch.from_numpy(anomaly_clip).to(torch.float32)
        x = self.interpolate(x, x.shape[0])
        # x = x.reshape(1, self.bags, self.frames//self.bags, -1).squeeze(0)
        x = x.squeeze(0)
        # x = torch.mean(x, dim=1)
        # 2 means labelled
        return x, 2
    


class NormalVideo_modified(Dataset):
    '''
        Custom dataset for Normal videos 
    '''
    def __init__(
            self,
            base_dir,
            path_to_unlabelled,
            normal_idx,
            frames = 3000, 
    ):
        self.base_dir = base_dir
        self.path_to_unlabelled = path_to_unlabelled
        self.frames = frames
        self.normal_idx = normal_idx
        
        with open('normal_train.txt', 'r') as f:
            self.list_of_videos = f.read().splitlines()

        self.labelled_list = [os.path.join(self.base_dir, vid) for vid in  self.list_of_videos]

        self.list_of_unlabelled_videos = os.listdir(self.path_to_unlabelled)
        self.list_unlabelled_normal = [self.list_of_unlabelled_videos[i] for i in  self.normal_idx]  
        self.unlabelled_list = [os.path.join(self.path_to_unlabelled, vid) for vid in  self.list_unlabelled_normal]

        self.total_list = self.labelled_list + self.unlabelled_list

    def __len__(self):
        return len(self.total_list)
    
    
    def interpolate(self, x, input_frames):
        # adding a batch
        x = x.unsqueeze(0)
        # [bs, frames, 512]

        x = x.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.frames, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)

        return x

        
    def __getitem__(self, idx: int):
        path = os.path.join(self.base_dir, self.total_list[idx])
        anomaly_clip = np.load(path)
        x = torch.from_numpy(anomaly_clip).to(torch.float32)
        x = self.interpolate(x, x.shape[0])
        # x = x.reshape(1, self.bags, self.frames//self.bags, -1).squeeze(0)
        x = x.squeeze(0)
        # x = torch.mean(x, dim=1)
        # -2 means unlabelled
        if 'normal' in path.lower():
            return x, 2
        else:
            return x, -2
    


class AnomalyVideo(Dataset):
    '''
        Custom dataset for Anomaly videos 
    '''
    def __init__(
            self,
            base_dir,
            frames = 3000, 

    ):
        self.base_dir = base_dir
        self.frames = frames
        
        with open('anomaly_train.txt', 'r') as f:
            self.list_of_videos = f.read().splitlines()

    def __len__(self):
        return len(self.list_of_videos)
    
    
    def interpolate(self, x, input_frames):
        # adding a batch
        x = x.unsqueeze(0)
        # [bs, frames, 512]

        x = x.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.frames, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)

        return x
    
    def __getitem__(self, idx: int):
        anomaly_clip = np.load(os.path.join(self.base_dir, self.list_of_videos[idx]))
        x = torch.from_numpy(anomaly_clip).to(torch.float32)
        x = self.interpolate(x, x.shape[0])
        # x = x.reshape(1, self.bags, self.frames//self.bags, -1).squeeze(0)
        # x = torch.mean(x, dim=1)
        x = x.squeeze(0)
        return x, 2





class AnomalyVideo_modified(Dataset):
    '''
        Custom dataset for Anomaly videos 
    '''
    def __init__(
            self,
            base_dir,
            path_to_unlabelled,
            anomaly_idx,
            frames = 3000, 
    ):
        self.base_dir = base_dir
        self.path_to_unlabelled = path_to_unlabelled
        self.frames = frames
        self.anomaly_idx = anomaly_idx
        
        with open('anomaly_train.txt', 'r') as f:
            self.list_of_videos = f.read().splitlines()

        self.labelled_list = [os.path.join(self.base_dir, vid) for vid in  self.list_of_videos]

        self.list_of_unlabelled_videos = os.listdir(self.path_to_unlabelled)
        self.list_unlabelled_anomaly = [self.list_of_unlabelled_videos[i] for i in  self.anomaly_idx]  
        self.unlabelled_list = [os.path.join(self.path_to_unlabelled, vid) for vid in  self.list_unlabelled_anomaly]

        self.total_list = self.labelled_list + self.unlabelled_list

    def __len__(self):
        return len(self.total_list)
    
    
    def interpolate(self, x, input_frames):
        # adding a batch
        x = x.unsqueeze(0)
        # [bs, frames, 512]

        x = x.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.frames, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)

        return x
    
    def __getitem__(self, idx: int):
        path = os.path.join(self.base_dir, self.total_list[idx])
        anomaly_clip = np.load(path)
        x = torch.from_numpy(anomaly_clip).to(torch.float32)
        x = self.interpolate(x, x.shape[0])
        # x = x.reshape(1, self.bags, self.frames//self.bags, -1).squeeze(0)
        # x = torch.mean(x, dim=1)
        x = x.squeeze(0)
        if 'anomaly' in path.lower():
            return x, 2
        else:
            return x, -2


    
class ValidationVideo(Dataset):
    def __init__(self, base_dir, frames=3000):
        self.base_dir = base_dir
        self.frames = frames

        with open('test_complete.txt', 'r') as f:
            self.list_of_videos = f.read().splitlines()
    
    def __len__(self):
        return len(self.list_of_videos)
    
    def interpolate(self, x, input_frames):
        # adding a batch
        x = x.unsqueeze(0)
        # [bs, frames, 512]

        x = x.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.frames, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)

        return x
    
    def __getitem__(self, idx):
        if 'Normal' in self.list_of_videos[idx]:
            name, num_frames, extra = self.list_of_videos[idx].split(' ')
            video_gts = int(extra[1])
            folder_name = 'Normal'
        else:
            name, num_frames, extra = self.list_of_videos[idx].split('|')
            video_gts = extra[1:-1].split(',')
            video_gts = [int(i) for i in video_gts]
            folder_name = 'Anomaly'
            

        clip_feat = np.load(os.path.join(self.base_dir, folder_name, name))
        x = torch.from_numpy(clip_feat).to(torch.float32)
        # interpolating it
        x = self.interpolate(x, x.shape[0])
        # removing the batch we don't need it; it would be added by the dataloader
        x = x.squeeze(0)

        return x, int(num_frames), video_gts


class UnlabelledVideo(Dataset):
    def __init__(self, base_dir, frames=3000):
        self.base_dir = base_dir
        self.frames = frames

        """with open('test_complete.txt', 'r') as f:
            self.list_of_videos = f.read().splitlines()"""
        
        self.list_of_videos = os.listdir(self.base_dir)
    
    def __len__(self):
        return len(self.list_of_videos)
    
    def interpolate(self, x, input_frames):
        # adding a batch
        x = x.unsqueeze(0)
        # [bs, frames, 512]

        x = x.permute(0, 2, 1).unsqueeze(-1) # adding a dummy dimension (1, 512, frames, 1)
        x = F.interpolate(x, size=(self.frames, 1), mode='bilinear')
        x = x.squeeze(-1).permute(0, 2, 1)

        return x
    
    def __getitem__(self, idx):
                   
        #num_frames = x.shape[0]
        clip_feat = np.load(os.path.join(self.base_dir, self.list_of_videos[idx]))
        x = torch.from_numpy(clip_feat).to(torch.float32)
        num_frames = x.shape[0]
        # interpolating it
        x = self.interpolate(x, x.shape[0])
        # removing the batch we don't need it; it would be added by the dataloader
        x = x.squeeze(0)

        return x, int(num_frames)

# ds = ValidationVideo('D:\Git_Repo\CLIP_Videoframes_Dataloader')

# a, b, c = ds[26]

# print(c)

# print(type(b))



