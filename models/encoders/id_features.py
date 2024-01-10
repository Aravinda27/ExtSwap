import sys
sys.path.append("/content/drive/MyDrive/ID-disentanglement-Pytorch-master")
import torch

import os
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from Models.Encoders.face_encoder import ViT_face_ID_encoder
import torch.nn.functional as F
from PIL import Image

input_size=[112, 112]
embedding_size=512
model_path='/content/drive/MyDrive/Deepfakes/pre-trained_model/Face_transformer_model.pth'

def get_embeddings(image):
    '''
    id_transform=transforms.Compose([
          transforms.Resize(
              [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ), # smaller side resized
          transforms.CenterCrop([input_size[0], input_size[1]]),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      ],
    )
    '''
    
    #image=id_transform(image)
    #image=image.unsqueeze(0)
    image=F.interpolate(image,size=112)
    print("ID shape",image.shape)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out=ViT_face_ID_encoder(image_size=112,
            patch_size=8,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1)
    #out.to('cuda')
    out.load_state_dict(torch.load(model_path,map_location=torch.device("cpu")))
    out.to(device)
    out.eval()
    embeddings = np.zeros([1,embedding_size])
    with torch.no_grad():
      embeddings[:] = F.normalize(out(image.to(device))).cpu()
      embeddings=torch.tensor(embeddings)
      return embeddings
