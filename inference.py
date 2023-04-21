import torch
import numpy as np
import matplotlib.pyplot as plt
from models import CLIPFormer
from torch.utils.data import DataLoader
from clipDataset import *

# load model and do some inference
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_feats = 64
path_to_anomaly = '../Anomaly'
path_to_normal = '../Normal'
checkpoint = torch.load('./checkpoints/best_ssl_bce.pth')
model = CLIPFormer(num_layers=1, nhead=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
# training data
train_anomaly_ds = AnomalyVideo(path_to_anomaly, num_feats)
train_normal_ds = NormalVideo(path_to_normal, num_feats)

train_a_dl = DataLoader(train_anomaly_ds, batch_size=1, num_workers=16)
train_n_dl = DataLoader(train_normal_ds, batch_size=1, num_workers=16)


# performing some inference
def infer():
    model.eval()

    print('Performing evaluation on the model')
    # gts = []
    preds = []
    with torch.no_grad():
        for anomaly, normal in zip(train_a_dl, train_n_dl):
            clip_a, _ = anomaly
            clip_n, _ = normal

            # gt_a = [1 for _ in clip_a.shape[0]]
            # gt_n = [0 for _ in clip_n.shape[1]]

            # pass data to device
            clip_a, clip_n = clip_a.to(device), clip_n.to(device)

            score_a, score_n = model(clip_a), model(clip_n)

            score_a, score_n = score_a.squeeze(-1), score_n.squeeze(-1)

            pred_a, pred_n = torch.max(score_a, dim=1)[0], torch.max(score_n, dim=1)[0]

            combined_pred = torch.cat(pred_a, pred_n)

            combined_pred = combined_pred.cpu().detach().numpy()

            preds.extend(combined_pred)

    return preds


preds = infer()

plt.figure()
plt.hist(preds)
plt.savefig('figure.png')


            

            







    
