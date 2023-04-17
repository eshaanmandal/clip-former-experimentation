import torch
import torch.nn.functional as F


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

def bce(dataset, indexes, model, device='cpu',percent=0.1):
    total_loss = 0.0
    total_len = len(indexes[0])
    a_idxs, n_idxs = indexes[0], indexes[1]
    with torch.no_grad():
        for a, n in zip(a_idxs, n_idxs):
            # move data to gpu/cpu
            print(dataset[a_idx])
            a_clip, n_clip = dataset[a_idx].to(device), dataset[n_idx].to(device)

            # get predections
            a_pred, n_pred = model(a_clip), model(n_clip)

            # removing the useless dimesnion
            a_pred, n_pred = a_pred.squeeze(-1), n_pred.squeeze(-1)

            # Lets start with video that are normal

            ground_truth_normal = torch.zeros_like(n_pred)
            bce_for_normal += F.binary_cross_entropy(n_pred, ground_truth_normal)

            #For anomaly videos (a bit complex)

            # sort the predctions
            topk = int(percent * a_pred.shape[0])

            # sorting the tensor
            a_pred = torch.sort(a_pred, descending=True)

            anomalous_part = a_pred[:topk]
            normal_part = a_pred[topk:]

            # bce for normal part
            bce_normal_part = F.binary_cross_entropy(normal_part, torch.zeros_like(normal_part))

            # bce for anomalous part
            bce_anomalous_part = F.binary_cross_entropy(anomalous_part, torch.ones_like(anomalous_part))

            # a = kn
            k = bce_anomalous_part / (bce_normal_part + 1e-3)

            # normalizing rhe scores

            bce_normal_part *= k

            bce_for_anomalous = bce_anomalous_part + bce_normal_part

            total_loss += (bce_for_anomalous + bce_for_normal)
    
    return total_loss / total_len


        


