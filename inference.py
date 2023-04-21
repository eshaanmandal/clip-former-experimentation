import torch
import numpy as np
import matplotlib.pyplot as plt
from models import CLIPFormer
from torch.utils.data import DataLoader
from clipDataset import *
from tqdm import tqdm

# load model and do some inference
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_feats = 64
path_to_anomaly = '../Anomaly'
path_to_normal = '../Normal'
path_to_val = '..'

checkpoint = torch.load('./checkpoints/best_ssl_bce.pth')
model = CLIPFormer(num_layers=1, nhead=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
# training data
train_anomaly_ds = AnomalyVideo(path_to_anomaly, num_feats)
train_normal_ds = NormalVideo(path_to_normal, num_feats)

train_a_dl = DataLoader(train_anomaly_ds, batch_size=32, num_workers=16, shuffle=True)
train_n_dl = DataLoader(train_normal_ds, batch_size=32, num_workers=16, shuffle=True)


# validation data
valid_ds = ValidationVideo(path_to_val, num_feats)
val_dl = DataLoader(valid_ds, batch_size=1, shuffle=True, num_workers=16)

# performing some inference
def infer(data):
    model.eval()

    if data == 'train':
        print('Performing evaluation on the model')
        gts = []
        preds = []
        with torch.no_grad():
            for anomaly, normal in tqdm(zip(train_a_dl, train_n_dl), total=min(len(train_a_dl), len(train_n_dl))):
                clip_a, _ = anomaly
                clip_n, _ = normal

                gt_a = [1 for _ in range(clip_a.shape[0])]
                gt_n = [0 for _ in range(clip_n.shape[1])]

                gts.extend(gt_a)
                gts.extend(gt_n)

                # pass data to device
                clip_a, clip_n = clip_a.to(device), clip_n.to(device)

                score_a, score_n = model(clip_a), model(clip_n)

                score_a, score_n = score_a.squeeze(-1), score_n.squeeze(-1)

                pred_a, pred_n = torch.max(score_a, dim=1)[0], torch.max(score_n, dim=1)[0]

                combined_pred = torch.cat((pred_a, pred_n))

                combined_pred = combined_pred.cpu().detach().numpy()

                preds.extend(combined_pred)

        return gts, preds

    elif data == 'valid':
        preds = []
        with torch.no_grad():
            for clip_fts, _, _ in tqdm(val_dl):
                clip_fts = clip_fts.to(device)

                scores = model(clip_fts)

                scores = scores.squeeze(-1)

                pred = torch.max(scores, dim=1)[0]

                pred = pred.cpu().detach().numpy()

                preds.extend(pred)

        return preds










gts, preds = infer(data='train')

plt.figure()
plt.hist(gts, alpha=0.5, label="Ground truth")
plt.hist(preds, alpha=0.5, label="Predictions")
plt.legend(['Ground truth', 'Predictions'])
plt.title('Training set preds')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.savefig('figure.png')


            

            







    
