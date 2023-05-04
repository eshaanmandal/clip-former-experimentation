import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from tqdm import tqdm
import time
torch.multiprocessing.set_sharing_strategy('file_system')

# MIL loss function
# ***Version - 1 ****
def MIL(combined_anomaly_scores):
    '''
    MIL stands for Multiple Instance Loss
    It takes the maximum of two bags (here each bag has 3000 frames each)
    and tries to maximize the difference betwen them
    A perfect MIL should give anomaly video a score of 1 and normal video a score of 0
    
    loss = relu(1 - 1 + 0) = 0 # ideal case
    worst case = relu(1 - 0 + 1) = 2 # identifies normal video as anomaly and vice-versa
    '''
    combined_anomaly_scores = combined_anomaly_scores.squeeze(-1)
    total = combined_anomaly_scores.shape[0] # total number of combineed features 50% is from normal bag 50% is from anomaly bag
    offset = (total // 2)
    loss = 0 
    for i in range(offset):
        y_abnormal_max = torch.max(combined_anomaly_scores[i, :])
        y_normal_max = torch.max(combined_anomaly_scores[i+offset, :]) 
        loss += F.relu(1. - y_abnormal_max + y_normal_max) 
    # print(torch.sum(torch.square(torch.diff(combined_anomaly_scores[0:offset, :], dim=1)))
    consecutive_diff = torch.diff(combined_anomaly_scores[:offset, :], dim=1)
    squared_diff = torch.square(consecutive_diff)
    # print(consecutive_diff.shape, squared_diff.shape)
    smoothness_term = torch.sum(torch.sum(squared_diff,dim=1), dim=0)
    sparisty_term = torch.sum(torch.sum(combined_anomaly_scores[:offset, :], dim=1), dim=0)


    l1 = l2 = 1

    return (loss/offset) + l1*(smoothness_term/offset) + l2*(sparisty_term/offset)



def bce(score, feat_type, percentage=0.40):
    # 0 means normal videos
    if feat_type == 0:
        ground_truth_normal = torch.zeros_like(score)
        bce_loss = F.binary_cross_entropy(score, ground_truth_normal)
        return bce_loss

    # 1 means anomaly
    if feat_type == 1:
        ground_truth_anomalous = torch.zeros_like(score)
        topk = int(percentage * score.shape[0])
        rest = score.shape[1] - topk

        score = torch.sort(score, descending=True)[0]

        a_part = score[:topk]
        n_part = score[topk:]

        bce_normal_part = F.binary_cross_entropy(n_part, torch.zeros_like(n_part))

        bce_anomalous_part = F.binary_cross_entropy(a_part, torch.ones_like(a_part))

        bce_loss = (bce_anomalous_part/topk) + (bce_normal_part / rest)
        return bce_loss






            



        


