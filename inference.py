import torch
import numpy as np
import matplotlib.pyplot as plt
from models import CLIPFormer
from torch.utils.data import DataLoader
from clipDataset import *
from tqdm import tqdm
import seaborn as sn

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
def infer(data, frames=3000):
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
                gt_n = [0 for _ in range(clip_n.shape[0])]

                gts.extend(gt_a + gt_n)

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
        all_gts = []
        all_scores = []

        with torch.no_grad():
            for clip_fts, num_frames, video_gt in tqdm(val_dl):
                clip_fts, num_frames = clip_fts.to(device), num_frames.item()
                # calculating anomaly scores
                scores = model(clip_fts)
                scores = scores.squeeze(0).squeeze(-1) # scores dimension = (frames) 
                scores = scores.cpu().detach().numpy()  # transfer data to cpu to detach extra things in tensor and make it a numpy array  

                
                if num_frames > frames:
                    scores = [score for score in scores for _ in range(num_frames//frames)]
                    all_scores.extend(scores)
            
                last_score = scores[-1]
                remainder = num_frames % frames
                
                if remainder != 0:
                    all_scores.extend(last_score for _ in range(remainder))
                
                # Now turn for the ground truths

                video_gt_list = torch.zeros(num_frames)

                # for normal video the video_gts is 1 ; a string of len 1 is returned for normal videos
                if len(video_gt) == 1:
                    all_gts.extend(video_gt_list)
                else:
                    video_gt_list[video_gt[0]:video_gt[1]] = 1
                    try:
                        video_gt_list[video_gt[2]:video_gt[3]] = 1
                    except:
                        pass
                    all_gts.extend(video_gt_list)
        return all_gts, all_scores






gts, preds = infer(data='valid', frames=num_feats)
print(len(gts), len(preds))

plt.figure()
# plt.hist(preds, alpha=0.5, color='blue')
# plt.hist(gts, alpha=0.5, color='green')
plt.hist(preds, histtype='step', color='blue')
plt.hist(gts, histtype='step', color='green')
plt.yticks(np.arange(0, 1e6, 10000))
plt.grid()

plt.legend(["Predicted", "Ground truth"])

plt.savefig('figure#1.png')


            

            







    
