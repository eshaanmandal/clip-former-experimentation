import torch
from clipDataset import *
from models import CLIPFormer
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_feats = 64
path_to_val = '..'


checkpoint = torch.load('./checkpoints/best_ssl_bce.pth')
model = CLIPFormer(num_layers=1, nhead=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])


# validation data
valid_ds = ValidationVideo(path_to_val, num_feats)
val_dl = DataLoader(valid_ds, batch_size=1, shuffle=True, num_workers=16)

def plot():
    model.eval()

    with torch.no_grad():
        for i, (clip_fts, num_frames, video_gt) in tqdm(enumerate(val_dl)):
            clip_fts, num_frames = clip_fts.to(device), num_frames.item()
            cores = model(clip_fts)
            scores = scores.squeeze(0).squeeze(-1) # scores dimension = (frames) 
            scores = scores.cpu().detach().numpy()

            if num_frames > num_feats:
                scores = [score for score in scores for _ in range(num_frames // num_feats)]

            last_score = scores[-1]
            remainder = num_frames % num_feats

            if remainder != 0:
                scores.extend([last_score for _ in remainder])

            print(len(scores), num_frames)

plot()

            



