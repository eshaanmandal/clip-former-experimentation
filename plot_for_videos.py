import torch
from clipDataset import *
from models import CLIPFormer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_feats = 64
path_to_val = '..'


checkpoint = torch.load('./checkpoints/best_ssl_bce.pth')
model = CLIPFormer(num_layers=1, nhead=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])


# validation data
valid_ds = ValidationVideo(path_to_val, num_feats)
val_dl = DataLoader(valid_ds, batch_size=1, shuffle=True, num_workers=16)

save_dir = './plots'
pred_dict = {}

def plot():
    model.eval()

    with torch.no_grad():
        for (clip_fts, num_frames, video_gt, name) in tqdm(val_dl):
            name = name[0]
            clip_fts, num_frames = clip_fts.to(device), num_frames.item()
            scores = model(clip_fts)
            scores = scores.squeeze(0).squeeze(-1) # scores dimension = (frames) 
            scores = scores.cpu().detach().numpy()

            if num_frames > num_feats:
                scores = [score for score in scores for _ in range(num_frames // num_feats)]

            last_score = scores[-1]
            remainder = num_frames % num_feats

            if remainder != 0:
                scores.extend([last_score for _ in range(remainder)])

            video_gt_list = torch.zeros(num_frames)

            if len(video_gt) == 1:
                # video_gt_list = video_gt_list.cpu().detach().numpy()
                continue
            else:
                video_gt_list[video_gt[0]:video_gt[1]] = 1
                try:
                    video_gt_list[video_gt[2]:video_gt[3]] = 1
                except:
                    pass
                video_gt_list = video_gt_list.cpu().detach().numpy()

                # plotting the frames
                pred_dict[name] = scores



                plt.figure()
                plt.plot(scores, label='predicted', color='green')
                plt.plot(video_gt_list, label='ground truth', color='blue')
                plt.ylim([0, 2])
                plt.ylabel('Score')
                plt.xlabel('Frames')
                plt.legend(["Prediction", "Ground truth"])
                plt.savefig(f"./plots/{name.split('.')[0]}.png", dpi=1200)
                plt.close()
    with open('predictions_on_val.pkl', 'wb') as f:
        pickle.dump(pred_dict, f)

                

plot()

            



