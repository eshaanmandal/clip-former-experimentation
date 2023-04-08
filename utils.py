import torch
import torch.nn.functional as F


# MIL loss function
# ***Version - 1 ****
def MIL(combined_anomaly_scores, batch_size):
    '''
    MIL stands for Multiple Instance Loss
    It takes the maximum of two bags (here each bag has 3000 frames each)
    and tries to maximize the difference betwen them
    A perfect MIL should give anomaly video a score of 1 and normal video a score of 0
    
    loss = relu(1 - 1 + 0) = 0 # ideal case
    worst case = relu(1 - 0 + 1) = 2 # identifies normal video as anomaly and vice-versa
    '''
    combined_anomaly_scores = combined_anomaly_scores.squeeze(-1) 
    loss = 0 
    for i in range(batch_size):
        y_abnormal_max = torch.max(combined_anomaly_scores[i, :])
        y_normal_max = torch.max(combined_anomaly_scores[i+batch_size, :]) 
        loss += F.relu(1. - y_abnormal_max + y_normal_max) 
    return loss/batch_size

    # combined loss = (2*batch_size)
    #loss = 0
    #for i in range(batch_size):
    #    abnormal_score = combined_anomaly_scores[i]
    #    normal_score = combined_anomaly_scores[i + batch_size]
    #    loss += F.relu(1 - abnormal_score + normal_score)
    #return loss / batch_size
