import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor



class FCNNLayer(nn.Sequential):
    def __init__(self, emb_size=512, output_classes=1):
        super().__init__(
            nn.Linear(emb_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

class Embeddings(nn.Module):
    def __init__(self, sequence_len = 3000, emb_size=512, heads=8):
        super().__init__()
        # cls_token 
        #self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        #self.position = nn.Parameter(torch.randn(sequence_len+1, emb_size))

    def forward(self, x):
        # batches
        #b, _, _ = x.shape
        #cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        #x = torch.cat([cls_token, x], dim=1)
        #print(self.position.shape)
        #x += self.position
        return x
    
class CLIPFormer(nn.Sequential):
    def __init__(
            self,
            num_layers=1,
            nhead=8,
            emb_size=512,
            sequence_len=3000
    ):
        encoder = nn.TransformerEncoderLayer(emb_size, nhead)
        super().__init__(
            #Embeddings(emb_size = emb_size),
            nn.TransformerEncoder(encoder, num_layers),
            FCNNLayer()
        )




# print(summary(CLIPFormer(), (3000, 512), device='cpu'))

