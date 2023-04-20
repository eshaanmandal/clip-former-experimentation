from clipDataset import *
from models import CLIPFormer
import torch
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from utils import *
from sklearn import metrics
import random
import wandb
import argparse
import gc
import torchmetrics


# these signify the labelled and unlabelled video
LABELLED = 2
UNLABELLED = -2 

def train(
        epoch, 
        ssl_step, 
        model, 
        optim, 
        device, 
        train_dl, 
        batch_size, 
        unlabelled_dl, 
        valid_bs, 
        percent_data, 
        num_feats, 
        unlabelled_ds, 
        indexes
):
    '''
        Method for training the CLIP-ANOMALY-DETECTION Model

    '''
    model.train() # sets the model into training mode
    print(f'Epoch {epoch+1}')
    # get the index
    
    anomaly_dl, normal_dl = train_dl[0], train_dl[1] # train_dl is a list [compactly passing the anomaly and normal clip feats dataloader]
    running_loss = 0.0
    l = min(len(anomaly_dl), len(normal_dl)) # simply means len(either dataloader) because same length for both
    for anomaly, normal in tqdm(zip(anomaly_dl, normal_dl), total=l):
        clip_a, label_a = anomaly
        clip_n, label_n = normal

        clip_a, clip_n = clip_a.to(device), clip_n.to(device)

        # calculating losses
        score_a, score_n = model(clip_a), model(clip_n)

        
        # cocatenating the anomaly and normal scores along the batch dimension [useful for computing MIL loss]
        combined_anomaly_scores = torch.cat([score_a, score_n], dim=0)
        loss = MIL(combined_anomaly_scores)

        if ssl_step != 0:
            try:
                loss += sum([bce(score_a[i], 1) for i in range(label_a.item()) if label_a[i] == -2]) \
                    / torch.numel(label_a[label_a == -2])
            except:
                loss += 0 
            try:
                loss += sum([bce(score_n[i], 0) for i in range(label_n.item()) if label_n[i] == -2]) \
                    / torch.numel(label_n[label_n == -2 ])
            except:
                loss += 0

        # if ssl_step != 0:
        #     count = 1
        #     for i in range(label_a.item()):
        #         count = 0 
        #         if label_a[i] == -1:
        #             loss += bce(score_a[i], 1)
        #             count += 1
        

        #     for i in range(label_n.item()):
        #         count = 0 
        #         if label_n[i] == -1:
        #             loss += bce(score_n[i], 0)
        #             count+=1
        #     loss /= count
        
        running_loss += loss.item()



        optim.zero_grad()
        loss.backward()
        optim.step()

    print('train loss = {}'.format(running_loss/l))
    del anomaly_dl, normal_dl, clip_a, clip_n, score_a, score_n
    gc.collect()
    torch.cuda.empty_cache()
    return (running_loss / l)


def test(epoch, model, optim, device, val_dl, batch_size, frames=3000):
    '''
        Method for doing test/validation on the testing clip feats.
    '''
    print(f'Performing evaluation of epoch {epoch+1}')
    model.eval() # Sets the model to eval mode
    all_gts = []
    all_scores = []

    with torch.no_grad():
        for clip_fts, num_frames, video_gt in tqdm(val_dl):
            clip_fts, num_frames = clip_fts.to(device), num_frames.item()
            # calculating anomaly scores
            scores = model(clip_fts)
            scores = scores.squeeze(0).squeeze(-1) # scores dimension = (frames) 
            scores = scores.cpu().detach().numpy()  # transfer data to cpu to detach extra things in tensor and make it a numpy array  

            '''
                The important part

                Since, we have to make framewise prediction but, we our clip features is working on a reduced number of frames 
                (3000 for e.g.). What we need to do is extend the features so as to make it equal to the number of features. 

                Case 1:
                lets say we have a prediction sequence of 3000 but 9600 frames (of the video on which we are trying to make preds)

                So our 3000 feats have to fit 9600 frames

                Therefore, we repeat each feature 9600 //  3000 = 3 ; we repeat each feature 3 time and for the remainder we extend the
                scores[-1] the last prediction (of the 3000 frames)

                Case 2: if the number of frames is less is number than the predicted features

                Just fill the prediction with the last score of (3000 preds) [times the number of frames]
                lets day we have 100 frames so, we take scores[-1] (lets say 0.6) and make an array [0.6, 0.6, 0.6, ..... 0.6] (len 100)
                 
            '''
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

    all_scores, all_gts = torch.tensor(all_scores), torch.tensor(all_gts)
    auroc = torchmetrics.AUROC(task="binary")
    auc = auroc(all_scores, all_gts)
    return auc
            
            

def seed_everything(seed: int = 10):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_once(
    train_anomaly_dl, 
    train_normal_dl, 
    val_dl,
    train_bs, 
    valid_bs, 
    epochs, 
    ssl_step, 
    num_feats, 
    best_auc, 
    model_name, 
    unlabelled_dl,
    percent_data, 
    unlabelled_ds,
    indexes
):
    #aucs = []

    for epoch in range(epochs):
        loss = train(
            epoch,
            ssl_step,
            model, 
            optimizer, 
            device, 
            [train_anomaly_dl, train_normal_dl], 
            train_bs, unlabelled_dl, valid_bs, 
            percent_data, 
            num_feats,
            unlabelled_ds,
            indexes
        )
        #wandb.log({"train_loss":loss})
        auc = test(epoch, model, optimizer, device, val_dl, valid_bs, num_feats)
        #aucs.append(auc)
        print(f'AUC score for epoch {epoch+1} : {auc}')
        if auc > best_auc:
            save_path = os.path.join('./checkpoints',model_name)
            if args.save:
                print(f'Saving the best model params to : {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'auc':auc
                }, save_path)
            best_auc = auc
        print(f'Best AUC score is {best_auc}')

    #path_for_plots = './plots/bs_2_epochs_10.png'
    #plt.figure()
    #plt.plot(list(range(1, epochs+1)), aucs)
    #plt.xlabel('epochs')
    #plt.ylabel('AUC')
    #plt.savefig(path_for_plots)
    
    return best_auc, model

def predict_on_unlabelled(model, device, unlabelled_dl, valid_bs, percent_data = 0.1, frames=3000):
    
    # print(f'Getting prediction on unlabelled dataset:')
    model.eval() # Sets the model to eval mode
    predicted_gts = []  #list of predicted video level labels for all unlabelled videos

    with torch.no_grad():
        for clip_fts, num_frames in tqdm(unlabelled_dl):

            clip_fts, num_frames = clip_fts.to(device), num_frames.item()
            # calculating anomaly scores
            scores = model(clip_fts)
            scores = scores.squeeze(0).squeeze(-1) # scores dimension = (frames) 
            scores = scores.cpu().detach().numpy() 

            output = max(scores)
            predicted_gts.append(output)

    sorted_idx = np.argsort(np.array(predicted_gts))
    num_vids = int(percent_data*len(predicted_gts))

    normal_idx = [sorted_idx[i] for i in range(num_vids)]
    anomaly_idx = [sorted_idx[-i-1] for i in range(num_vids)]

    # print(len(normal_idx), len(anomaly_idx))
    return normal_idx, anomaly_idx


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate the clip anomaly transformer')
    parser.add_argument('--seed', type=int, default=None, help='Provide seed for reproducability')
    parser.add_argument('--feat', type=int, default=64, help='Interpolation number of frames')
    parser.add_argument('--layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--wd', type=float, default=0.001, help='Weight decay of the Adam optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate of the Adam optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs till which to train the model once')
    parser.add_argument('--total_epochs', type=int, default=2, help='Number of epochs for which to train the whole model including unlabelled dataset')
    parser.add_argument('--n_path', type=str, default='../Normal', help='Path to normal video clip feats.')
    parser.add_argument('--a_path', type=str, default='../Anomaly', help='Path to anomaly video clip feats.')
    parser.add_argument('--val_path', type=str, default='..', help='Path to the validation videos')
    parser.add_argument('--unlabelled_path', type=str, default='..', help='Path to the unalbelled video clip features')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use wandb')
    parser.add_argument('--device', type=str, default='cpu', help='The device to use for model training and inference')
    parser.add_argument('--train_bs', type=int, default=1, help='Training batch size')
    parser.add_argument('--p_name', type=str, default='default_proj', help='name of the wandb project (if using wandb)')
    parser.add_argument('--save', type=bool, default=False, help='Save the best AUC score model')
    parser.add_argument('--percent_data', type=float, default=0.2, help='percentage of unlabelled data to use for total normal and abnormal (write as 0.1 or 0.2)')
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='Save name of the model params')
    args = parser.parse_args()

    if not os.path.isdir('./checkpoints'):
        print('The directory for checkpoints doesnt exist creating one at ./checkpoints ')
        os.makedirs('./checkpoints')
    else:
        print('The directory already exists skipping the folder creation step')



    # the hyperparameters
    if args.seed is not None:
        seed_everything(int(args.seed))
    seed = args.seed
    num_feats = args.feat
    device = args.device
    num_layers = args.layers
    heads = args.heads
    model = CLIPFormer(num_layers=num_layers, nhead=heads).to(device)
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    total_epochs = args.total_epochs
    percent_data = args.percent_data
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Initialising the 
    path_to_anomaly = args.a_path
    path_to_normal = args.n_path

    path_to_val = args.val_path

    path_to_unlabelled = args.unlabelled_path

    train_bs = args.train_bs
    valid_bs = 1 # Keeping validation batch size fixed
    
    print(f'Device: {device}\nLearning rate: {lr}\nWeight decay: {wd}\n Epochs: {epochs}\nSeed: {seed}\n')
    # datasets
    train_anomaly_ds = AnomalyVideo(path_to_anomaly, num_feats)
    train_normal_ds = NormalVideo(path_to_normal, num_feats)

    valid_ds = ValidationVideo(path_to_val, num_feats)

    unlabelled_ds = UnlabelledVideo(path_to_unlabelled, num_feats)

    # dataloaders

    train_anomaly_dl = DataLoader(train_anomaly_ds, batch_size=train_bs, shuffle=True, num_workers=16)
    train_normal_dl = DataLoader(train_normal_ds, batch_size=train_bs, shuffle=True, num_workers=16)

    val_dl = DataLoader(valid_ds, batch_size=valid_bs, shuffle=True, num_workers=16)

    unlabelled_dl = DataLoader(unlabelled_ds, batch_size=valid_bs, shuffle=True, num_workers=16) 

    if args.use_wandb:
        wandb.init(
                project=args.p_name,
                config={
                    "learning_rate":lr,
                    "architecture": "Transformer",
                    "dataset": "UCF Crime",
                    "epochs" : epochs,
                    "train_batch_size":train_bs,
                    "validation_batch_size":valid_bs,
                    "encoder_layers": num_layers,
                    "seed": args.seed,
                } 
        )

    
    list_of_unlabelled_videos = os.listdir(path_to_unlabelled)
    curr_best_auc = 0.5
    indexes=None
    for ssl_step in range(total_epochs):

        print("################################")
        print("SSL Step: ", ssl_step+1)
        print("################################")

        print("Length of normal dataset: ", len(train_normal_ds))
        print("Length of anomaly dataset: ", len(train_anomaly_ds))
        print("Length of unlabelled dataset remaining: ", len(list_of_unlabelled_videos))

        best_auc, model = train_once(
            train_anomaly_dl, 
            train_normal_dl, 
            val_dl, 
            train_bs, 
            valid_bs, 
            epochs, 
            ssl_step,
            num_feats, 
            curr_best_auc, 
            args.model_name,
            unlabelled_dl,
            percent_data, 
            unlabelled_ds,
            indexes
        )
        print("BEST TEST AUC: ", best_auc.item())

        if best_auc > curr_best_auc:
            curr_best_auc = best_auc

        if args.use_wandb:
            wandb.log({"auc_per_run":curr_best_auc})
        
        ## Do we need to load best model??? YES
        # if models are being saved load the best 
        if args.save:
            load_path = os.path.join('./checkpoints', args.model_name)
            print(f'Loading the best model params from : {load_path}')
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        # for that update later
        
        print(f'Getting prediction on unlabelled dataset:')
        normal_idx, anomaly_idx = predict_on_unlabelled(model, device, unlabelled_dl, valid_bs, percent_data/2, num_feats)
        indexes = [anomaly_idx, normal_idx]
        
        ######
        """
        Got predictions acc to the order on unlabelled video dl
        Now, you've to incorporate those in train dataloader - the method will be slightly different
        """
        ######

        # modifying the datasets

        train_anomaly_ds = AnomalyVideo_modified(path_to_anomaly, path_to_unlabelled, anomaly_idx, num_feats)
        train_normal_ds = NormalVideo_modified(path_to_normal, path_to_unlabelled, normal_idx, num_feats)

        # modifying the dataloaders

        train_anomaly_dl = DataLoader(train_anomaly_ds, batch_size=train_bs, shuffle=True,num_workers=16)
        train_normal_dl = DataLoader(train_normal_ds, batch_size=train_bs, shuffle=True, num_workers=16)
    if args.use_wandb:
        wandb.finish()



        # list_of_unlabelled_videos = [list_of_unlabelled_videos[idx] for idx in unlabelled_idx]
