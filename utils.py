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
    return loss/offset

    # combined loss = (2*batch_size)
    #loss = 0
    #for i in range(batch_size):
    #    abnormal_score = combined_anomaly_scores[i]
    #    normal_score = combined_anomaly_scores[i + batch_size]
    #    loss += F.relu(1 - abnormal_score + normal_score)
    #return loss / batch_size

def bce_new(dataset,indexes, model, device='cpu', percentage=0.5):
    # start = time.time()
    model.eval()
    total_loss = 0.0
    total_len = len(indexes[0])
    a_idxs, n_idxs = indexes[0], indexes[1]

    # making a custom subset sampler
    #anomaly_sampler = SubsetRandomSampler(a_idxs)
    #normal_sampler = SubsetRandomSampler(n_idxs)

    #subset of data
    a = Subset(dataset, a_idxs)
    n = Subset(dataset, n_idxs)

    #print(len(a_idxs), len(n_idxs))

    # Defining a dataloader
    a_dl = DataLoader(a, shuffle=True, batch_size=32, num_workers=16)
    n_dl = DataLoader(n, shuffle=True, batch_size=32, num_workers=16)

    with torch.no_grad():
        for anomaly, normal in zip(a_dl,n_dl):
            clip_a, clip_n = anomaly[0], normal[0]

            # moving the dataset to device
            clip_a, clip_n = clip_a.to(device), clip_n.to(device)

            #get some predictions
            score_a, score_n = model(clip_a), model(clip_n)

            # getting rid of the last dims
            score_a, score_n = score_a.squeeze(-1), score_n.squeeze(-1)

            # starting with pure normal videos 
            ground_truth_normal = torch.zeros_like(score_n)
            bce_for_normal = F.binary_cross_entropy(score_n, ground_truth_normal)


            # Now move to anomalous video
            # the case is bit complicated here
            # print(score_a.shape) # betting its batch size, feats

            ground_truth_anomalous = torch.zeros_like(score_a)
            
            topk = int(percentage * score_a.shape[1])
            rest = score_a.shape[1] - topk

            score_a = torch.sort(score_a, dim=1, descending=True)[0]

            a_part = score_a[:,:topk]
            n_part = score_n[:, topk:]

            # BCE for the normal part
            bce_normal_part = F.binary_cross_entropy(n_part, torch.zeros_like(n_part))

            # BCE for anomalous part
            bce_anomalous_part = F.binary_cross_entropy(a_part, torch.ones_like(a_part))

            # computing the coeff
            #k = bce_anomalous_part / (bce_normal_part + 1e-3)

            # normalizing the scores
            #bce_normal_part *= k
            bce_for_anomalous = (bce_anomalous_part/topk) + (bce_normal_part / rest)
            # bce_for_anomalous = F.binary_cross_entropy(score_a, ground_truth_anomalous)
            total_loss += (bce_for_anomalous + bce_for_normal)

    # end = time.time()
    # print(f'It took {end - start} s')

    return total_loss / total_len


def bce(score, feat_type, percentage=0.40):
    # 0 means normal videos
    if feat_type == 0:
        ground_truth_normal = torch.zeros_like(score)
        bce_loss = F.binary_cross_entropy(score, ground_truth_normal)
        return bce_loss

    # 1 means anomaly
    if feat_type == 1:
        ground_truth_anomalous = torch.zeros_like(score_a)
        topk = int(percentage * score.shape[0])
        rest = score_a.shape[1] - topk

        score = torch.sort(score, descending=True)[0]

        a_part = score[:topk]
        n_part = score[topk:]

        bce_normal_part = F.binary_cross_entropy(n_part, torch.zeros_like(n_part))

        bce_anomalous_part = F.binary_cross_entropy(a_part, torch.ones_like(a_part))

        bce_loss = (bce_anomalous_part/topk) + (bce_normal_part / rest)
        return bce_loss






            



        


